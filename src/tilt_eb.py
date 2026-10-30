"""Empirical-Bayes noise model for the public-label tilt (src/tilt.py).

tilt.py gives every scored file the same noise sd (one rounding step x noise_units). Files differ: near-base files
(own LB-blend probes, atlas probes) are predicted by the other files to < 1u, far notebook files carry a
linearisation error that grows with their distance to the base, and some notebook scores belong to another
version of the notebook than the output we downloaded (gross errors). Here the per-file noise variance is

    var_k = r2 + r2 * (k0 + k1 * M_kk / median(M_kk)) / w_k

r2 = variance of one 5-decimal rounding (the file's own, a floor), M_kk = prior variance of file k's public-score surprise (its size),
w_k = Student-t weight (EM; nu degrees of freedom) that down-weights files the others cannot explain.
(k0, k1) maximise the REML marginal likelihood of the surprises; the base's rounding is shared by all files.

    from tilt_eb import TiltEB
    t = TiltEB(p).fit(files, scores, base_score)      # builds g, h, M once
    t.fit_noise(nu=4)                                 # REML (k0, k1) + Student-t weights; then t.score(eta)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from tilt import Tilt


class TiltEB(Tilt):
    def __init__(self, p, f=0.2, nuisance=True):
        super().__init__(p, f=f, noise_units=1.0, nuisance=nuisance)
        self.r2 = self.sd ** 2  # variance of one rounding step in pair units (sd already / sqrt 12)
        self.noise_var = None
        self.w = None

    # --- linear algebra for a given noise vector ---------------------------------------------------------------
    def _noise_mat(self, nv):
        """Noise covariance of the files. Rounding is independent. The model error (nv - r2) is independent too unless
        kernel_power m > 0: then two files' errors correlate as corr(M)^m, so near-identical files share their model error
        and their DIFFERENCE is known to rounding precision (the second-order AUC terms of two similar files are similar)."""
        blocks = getattr(self, "noise_blocks", None)
        if blocks:  # files of one block (own nudge probes of one anchor) share their model error; rounding stays independent
            e = np.sqrt(np.maximum(nv - self.r2, 0))
            sn = np.diag(nv)
            for b in blocks:
                sn[np.ix_(b, b)] = np.outer(e[b], e[b])
                sn[b, b] = nv[b]
            return sn
        m_pow = getattr(self, "kernel_power", 0)
        if not m_pow:
            return np.diag(nv)
        d = np.sqrt(np.diag(self.m))
        corr = self.m / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        e = np.sqrt(np.maximum(nv - self.r2, 0))
        return np.diag(np.full(len(nv), self.r2)) + np.outer(e, e) * np.power(np.abs(corr), m_pow) * np.sign(corr) ** (m_pow % 2)

    def _cov(self, nv):
        k = len(self.yraw)
        return self.m + self._noise_mat(nv) + self.r2 * np.ones((k, k))

    def _x(self):
        cs, dist = self.cs, self.dist
        return np.stack([cs / np.abs(cs).max(), dist / max(dist.max(), 1e-12)], 1)

    def reml(self, nv):
        c = self._cov(nv)
        try:
            ch = np.linalg.cholesky(c)
        except np.linalg.LinAlgError:
            return -np.inf
        ci = np.linalg.inv(c)
        y = self.yraw
        if self.nuisance:
            x = self._x()
            a = x.T @ ci @ x
            beta = np.linalg.solve(a, x.T @ ci @ y)
            r = y - x @ beta
            ld_a = np.linalg.slogdet(a)[1]
        else:
            r, ld_a = y, 0.0
        return -0.5 * (2 * np.log(np.diag(ch)).sum() + ld_a + r @ ci @ r)

    def noise_vector(self, k0, k1, w=None):
        mk = np.diag(self.m)
        extra = self.r2 * (k0 + k1 * mk / np.median(mk))  # model error: heavy-tailed (Student-t weights)
        if getattr(self, "group", None) is not None and len(self.group) == len(mk):
            extra = extra * np.where(self.group, getattr(self, "gmult", 1.0), 1.0)  # suspect files: own level
        if w is not None:
            extra = extra / w
        return self.r2 + extra  # + the file's own rounding, a known floor

    def fit_noise(self, nu: float = 0.0, iters: int = 30, fixed=None, exact=None, verbose=True):
        """REML over (log k0, log k1); with nu > 0, Student-t EM weights on top (re-optimising k0, k1 each pass).
        exact: boolean mask of files whose scores are exact (own submissions): their weight is never below 1."""
        k = len(self.yraw)
        w = np.ones(k)
        exact = np.zeros(k, bool) if exact is None else np.asarray(exact, bool)

        def negll(th, w):
            return -self.reml(self.noise_vector(np.exp(th[0]), np.exp(th[1]), w))

        th = np.log([1.0, 1.0]) if fixed is None else np.log(fixed)
        grouped = getattr(self, "group", None) is not None and len(self.group) == k
        if grouped and fixed is None:
            th = np.append(th, np.log(getattr(self, "gmult", 4.0)))

        def negll_g(th, w):
            self.gmult = float(np.exp(th[2]))
            return -self.reml(self.noise_vector(np.exp(th[0]), np.exp(th[1]), w))

        for it in range(iters if nu > 0 else 1):
            if fixed is None:
                res = minimize(negll_g if grouped else negll, th, args=(w,), method="Nelder-Mead",
                               options={"xatol": 1e-3, "fatol": 1e-3})
                th = res.x
                if grouped:
                    self.gmult = float(np.exp(th[2]))
            nv = self.noise_vector(np.exp(th[0]), np.exp(th[1]), w)
            if nu <= 0:
                break
            # E-step: posterior of the noise of each file given all data, e_k ~ N(m_k, v_k)
            c = self._cov(nv)
            ci = np.linalg.inv(c)
            r = self._resid(ci)
            sn = self._noise_mat(nv)
            mvec = sn @ (ci @ r)
            vvec = np.diag(sn) - np.einsum("ij,ji->i", sn @ ci, sn)
            base = self.noise_vector(np.exp(th[0]), np.exp(th[1]))
            w_new = (nu + 1) / (nu + (mvec ** 2 + vvec) / base)
            w_new[exact] = np.maximum(w_new[exact], 1.0)
            if np.abs(np.log(w_new / w)).max() < 1e-3:
                w = w_new
                break
            w = w_new
        self.k0, self.k1 = np.exp(th[:2])
        self.w = w
        self.noise_var = self.noise_vector(self.k0, self.k1, w)
        self.solve_nv(self.noise_var)
        if verbose:
            print(f"REML noise: k0 {self.k0:.3g} k1 {self.k1:.3g} (x rounding var)"
                  + (f", suspect-group x{self.gmult:.3g}" if getattr(self, "group", None) is not None else "") + "; "
                  f"weights < 0.5: {(w < 0.5).sum()}, < 0.2: {(w < 0.2).sum()}; REML {self.reml(self.noise_var):.2f}")
        return self

    def _resid(self, ci):
        y = self.yraw
        if not self.nuisance:
            self.beta = np.zeros(2)
            return y
        x = self._x()
        self.beta = np.linalg.solve(x.T @ ci @ x, x.T @ ci @ y)
        return y - x @ self.beta

    def solve_nv(self, nv):
        c = self._cov(nv)
        q = np.linalg.inv(c)
        yv = self._resid(q)
        self.lam = q @ yv
        self.loo = self.lam / np.diag(q)
        self.loo_r2 = 1 - float(self.loo @ self.loo) / float(yv @ yv)
        gl = self.g.T.astype(np.float64) @ self.lam
        hl = self.h.T.astype(np.float64) @ self.lam
        self.da = self.va * gl + self.cab * hl
        self.db = self.cab * gl + self.vb * hl
        self.y = yv
        return self

    def solve(self):  # keep the parent's incremental add() working with the current noise model
        if self.noise_var is None or len(self.noise_var) != len(self.yraw):
            return super().solve()
        return self.solve_nv(self.noise_var)

    def subset(self, idx):
        """A copy restricted to the files idx (no rebuild: rows/columns of the K x K system are reused)."""
        idx = np.asarray(idx)
        t = object.__new__(type(self))
        t.__dict__.update(self.__dict__)
        t.g, t.h = self.g[idx], self.h[idx]
        t.yraw, t.cs, t.dist = self.yraw[idx], self.cs[idx], self.dist[idx]
        t.m = self.m[np.ix_(idx, idx)]
        t.noise_var, t.w = None, None
        if getattr(self, "group", None) is not None:
            t.group = self.group[idx]
        return t

    def file_coefs(self, s):
        w, u = self._wu(s)
        return (self.f * (w - self.wb)).astype(np.float32), (self.f * (u - self.ub)).astype(np.float32), \
            self.f ** 2 * (float(self.p @ w) - self.sb), float(np.square(w - self.wb).mean())

    def predict_dist(self, s, q=None):
        """Posterior predictive mean (public AUC) and sd (AUC units) of a NEW file's score, including its noise
        (rounding + model error at the fitted k0, k1 and weight 1). q = inverse covariance (cached by caller)."""
        g, h, c, d = self.file_coefs(s)
        va, vb, cab = (x.astype(np.float32) for x in (self.va, self.vb, self.cab))
        prior = float((g * va) @ g + (h * vb) @ h + 2 * (g * cab) @ h)
        kv = (self.g @ (va * g) + self.h @ (vb * h) + self.g @ (cab * h) + self.h @ (cab * g)).astype(np.float64)
        if q is None:
            q = np.linalg.inv(self._cov(self.noise_var))
        post = max(prior - kv @ q @ kv, 0.0)
        mk = np.median(np.diag(self.m))
        noise = self.r2 + self.r2 * (self.k0 + self.k1 * prior / mk)
        mean = self.predict(s)
        self.last_parts = (np.sqrt(post) / (self.n1 * self.n0), np.sqrt(noise) / (self.n1 * self.n0))
        return mean, np.sqrt(post + noise) / (self.n1 * self.n0)

    def lookahead(self, s, ys, eta=1.0, q=None):
        """One-step lookahead: if file s scored y (for each y in ys; AUC units, exact own score), the posterior-
        predicted public score of the re-fitted tilt at eta. The system is extended by one row/column analytically
        (noise model k0, k1, weights kept; the new file gets weight 1). Returns a list of predicted scores."""
        g, h, c, d = self.file_coefs(s)
        va, vb, cab = (x.astype(np.float32) for x in (self.va, self.vb, self.cab))
        prior = float((g * va) @ g + (h * vb) @ h + 2 * (g * cab) @ h)
        kv = (self.g @ (va * g) + self.h @ (vb * h) + self.g @ (cab * h) + self.h @ (cab * g)).astype(np.float64)
        k = len(self.yraw)
        m2 = np.block([[self.m, kv[:, None]], [kv[None, :], np.array([[prior]])]])
        mk = np.median(np.diag(self.m))
        nv2 = np.append(self.noise_var, self.r2 + self.r2 * (self.k0 + self.k1 * prior / mk))
        c2 = m2 + np.diag(nv2) + self.r2 * np.ones((k + 1, k + 1))
        q2 = np.linalg.inv(c2)
        cs2, dist2 = np.append(self.cs, c), np.append(self.dist, d)
        x2 = np.stack([cs2 / np.abs(cs2).max(), dist2 / max(dist2.max(), 1e-12)], 1) if self.nuisance else None
        out = []
        saved = (self.da, self.db, self.beta, self.cs, self.dist)
        for y in ys:
            yraw2 = np.append(self.yraw, self.n1 * self.n0 * (y - self.base_score) - c)
            if self.nuisance:
                beta = np.linalg.solve(x2.T @ q2 @ x2, x2.T @ q2 @ yraw2)
                r = yraw2 - x2 @ beta
            else:
                beta, r = np.zeros(2), yraw2
            lam = q2 @ r
            gl = self.g.T.astype(np.float64) @ lam[:k] + g.astype(np.float64) * lam[k]
            hl = self.h.T.astype(np.float64) @ lam[:k] + h.astype(np.float64) * lam[k]
            self.da = self.va * gl + self.cab * hl
            self.db = self.cab * gl + self.vb * hl
            self.beta, self.cs, self.dist = beta, cs2, dist2
            out.append(self.predict(self.score(eta)))
            self.da, self.db, self.beta, self.cs, self.dist = saved
        return out


class TiltGN(TiltEB):
    """One Gauss-Newton step for the bilinear AUC model: the files' coefficients are recomputed around the posterior
    mean (a_hat, b_hat) of a first fit instead of the prior (W = expected public NEGATIVES below a row uses b_hat,
    U = expected public POSITIVES above uses a_hat), the expected AUC change of each file becomes
    sum_i a_hat_i g'_i, and the prior stays N(a_bar, b_bar; V). With d0 = (da0, db0) the first fit's shift,
    y'_k = n1 n0 (A_k - A_b) - c'_k + G'_k d0 = G'_k delta + noise, delta ~ N(0, V)."""

    def __init__(self, p, ah, bh, da0, db0, f=0.2, nuisance=True):
        self.qw = np.clip(np.asarray(bh, np.float64), 0, None) / f
        self.pw = np.clip(np.asarray(ah, np.float64), 0, None) / f
        super().__init__(p, f=f, nuisance=nuisance)
        self.da0, self.db0 = np.asarray(da0, np.float64), np.asarray(db0, np.float64)
        self.sb = float(self.pw @ self.wb)

    def _wu(self, s):
        order = np.argsort(np.asarray(s, dtype=np.float64), kind="stable")
        q = self.qw[order]
        pp = self.pw[order]
        w = np.empty(self.n)
        u = np.empty(self.n)
        w[order] = np.cumsum(q) - q
        u[order] = pp.sum() - np.cumsum(pp)
        return w, u

    def add(self, files, scores):
        k = len(files)
        g = np.empty((k, self.n), dtype=np.float32)
        h = np.empty((k, self.n), dtype=np.float32)
        yv, cs, dist = np.empty(k), np.empty(k), np.empty(k)
        for i, s in enumerate(files):
            w, u = self._wu(s)
            gi, hi = self.f * (w - self.wb), self.f * (u - self.ub)
            g[i], h[i] = gi, hi
            cs[i] = self.f ** 2 * (float(self.pw @ w) - self.sb)
            dist[i] = float(np.square(w - self.wb).mean())
            yv[i] = self.n1 * self.n0 * (scores[i] - self.base_score) - cs[i] + float(self.da0 @ gi + self.db0 @ hi)
        va, vb, cab = (x.astype(np.float32) for x in (self.va, self.vb, self.cab))
        gv, hv, gc, hc = g * va, h * vb, g * cab, h * cab
        new = (gv @ g.T + hv @ h.T + gc @ h.T + hc @ g.T).astype(np.float64)
        cross = (gv @ self.g.T + hv @ self.h.T + gc @ self.h.T + hc @ self.g.T).astype(np.float64)
        self.m = np.block([[self.m, cross.T], [cross, new]])
        self.g, self.h = np.concatenate([self.g, g]), np.concatenate([self.h, h])
        self.yraw, self.cs, self.dist = (np.concatenate(x) for x in ((self.yraw, yv), (self.cs, cs), (self.dist, dist)))
        return self.solve()

    def predict(self, s):
        w, u = self._wu(s)
        g = self.f * (w - self.wb)
        h = self.f * (u - self.ub)
        c = self.f ** 2 * (float(self.pw @ w) - self.sb)
        y = float((self.da - self.da0) @ g + (self.db - self.db0) @ h)
        if self.nuisance:
            d = float(np.square(w - self.wb).mean())
            y += self.beta[0] * c / np.abs(self.cs).max() + self.beta[1] * d / max(self.dist.max(), 1e-12)
        return self.base_score + (y + c) / (self.n1 * self.n0)


def onesided_em(t, nu=2.0, exact=None, iters=6, pi0=0.2, sig_u=10.0, verbose=True, suspect=None, **kw):
    """One-sided score errors: a listed notebook score is its BEST version's, so when the file we hold is another
    version the listed score is too HIGH (never too low). Model r_k = e_k + noise, e_k = 0 (prob 1 - pi) or
    half-normal(sigma_e) (prob pi), r_k = file k's leave-one-out surprise residual. EM: subtract E[e_k | r_k] from
    each non-exact file's surprise, refit the robust tilt, re-estimate pi and sigma_e. Returns t (refitted)."""
    from scipy.stats import norm as _n
    k = len(t.yraw)
    exact = np.zeros(k, bool) if exact is None else np.asarray(exact, bool)
    y0 = t.yraw.copy()
    unit = 1e-5 * t.n1 * t.n0
    pi, se = pi0, sig_u * unit
    e = np.zeros(k)
    for it in range(iters):
        t.yraw = y0 - e
        t.fit_noise(nu=nu, exact=exact, verbose=False, **kw)
        q = np.linalg.inv(t._cov(t.noise_var))
        r = t.loo + e  # residual of the ORIGINAL surprise: what the others predict vs the listed score
        v = 1.0 / np.diag(q)
        m = r * se ** 2 / (se ** 2 + v)
        s = np.sqrt(se ** 2 * v / (se ** 2 + v))
        l1 = 2 * _n.pdf(r, 0, np.sqrt(se ** 2 + v)) * _n.cdf(m / s)
        l0 = _n.pdf(r, 0, np.sqrt(v))
        g = pi * l1 / np.maximum(pi * l1 + (1 - pi) * l0, 1e-300)
        g[exact] = 0
        if suspect is not None:  # only files whose listed score can belong to another version
            g[~np.asarray(suspect, bool)] = 0
        ee = m + s * _n.pdf(m / s) / np.maximum(_n.cdf(m / s), 1e-300)
        e = g * ee
        e2 = g * (ee ** 2 + s ** 2 - (m + s * 0) * 0)  # second moment approx (truncated var ignored)
        grp = ~exact if suspect is None else (np.asarray(suspect, bool) & ~exact)
        pi = float(np.clip(g[grp].mean(), 1e-3, 0.9))
        se = float(np.sqrt(max(e2[grp].sum() / max(g[grp].sum(), 1e-9), (1 * unit) ** 2)))
        if verbose:
            print(f"one-sided EM {it + 1}: pi {pi:.3f} sigma_e {se / unit:.1f}u, mean shift {e[~exact].mean() / unit:.2f}u, "
                  f"files with E[e] > 3u: {(e > 3 * unit).sum()}", flush=True)
    t.yraw = y0 - e
    t.fit_noise(nu=nu, exact=exact, verbose=False, **kw)
    t.onesided_e = e / unit
    return t
