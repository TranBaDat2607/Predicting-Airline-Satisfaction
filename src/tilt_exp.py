"""Exponential-tilt (mean-field) posterior for the public-label tilt.

src/tilt.py and src/tilt_eb.py put a Gaussian prior on (a_i, b_i), so a row's posterior is prior + linear shift. For the
rare state that is far outside its range (real fit: E[a] of low-p rows moves by +-43 % sd, 1.6 % of them go negative and
are clipped). Each row really has three states: public positive, public negative, private. Conditioning independent rows
on K linear measurements gives (Gibbs conditioning principle) an exponential tilt of each row's prior

    q_i(P) = abar_i exp(x_i) / D_i,  q_i(N) = bbar_i exp(z_i) / D_i,  q_i(X) = (1 - f) / D_i
    x_i = sum_k lam_k g_k(i),  z_i = sum_k lam_k h_k(i)

with lam the optimum of the convex dual

    sum_i log D_i(lam) - lam'(y - X beta + offset) + 1/2 lam' Sigma lam,   X' lam = 0 (flat prior on the nuisance beta).

Newton from lam = 0: the first step IS the Gaussian solution of tilt_eb. The tilted log-odds are linear in the features:

    logit q(positive | public) = logit p_i + eta (x_i - z_i)

relinearize() recomputes the bilinear AUC model's coefficients around the current q (the variational fixed point).

    t = TiltExp(p).fit(files, scores, base_score); t.fit_noise(nu=2)   # REML (k0, k1) + Student-t as tilt_eb
    t.fit_exp(em=3)                                                    # Newton + Student-t E-steps on the exact dual
    s = t.score_exp(1.0)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg.blas import ssyrk

from tilt_eb import TiltEB


class TiltExp(TiltEB):
    def _mv(self, lam):
        """x = G' lam, z = H' lam (float64 accumulation, one row of G at a time: no 8-byte copy of G)."""
        x, z = np.zeros(self.n), np.zeros(self.n)
        for k, v in enumerate(lam):
            x += v * self.g[k]
            z += v * self.h[k]
        return x, z

    def _q(self, x, z):
        # exp clipped far outside any useful range; D is normalised by the private state's constant
        ea = self.abar * np.exp(np.clip(x, -50, 50))
        eb = self.bbar * np.exp(np.clip(z, -50, 50))
        d = ea + eb + (1 - self.f)
        return ea / d, eb / d

    def _gmv(self, vp, vn):
        """G vp + H vn (K vector)."""
        vp32, vn32 = vp.astype(np.float32), vn.astype(np.float32)
        return (self.g @ vp32 + self.h @ vn32).astype(np.float64)

    def _hessian(self, qp, qn):
        vp, vn, c = qp * (1 - qp), qn * (1 - qn), -qp * qn
        s1 = np.sqrt(np.maximum(vp, 1e-30))
        c1 = c / s1
        s2 = np.sqrt(np.maximum(vn - c1 ** 2, 1e-30))
        s1, c1, s2 = (v.astype(np.float32) for v in (s1, c1, s2))
        # H = L1 L1' + L2 L2' with L1 = g s1 + h c1, L2 = h s2 (Cholesky of each row's 2 x 2 covariance)
        l1 = self.g * s1 + self.h * c1
        m = ssyrk(1.0, l1.T, trans=1, lower=0).astype(np.float64)
        del l1
        l2 = self.h * s2
        m += ssyrk(1.0, l2.T, trans=1, lower=0).astype(np.float64)
        del l2
        return m + np.triu(m, 1).T

    def _sigma(self, nv):
        k = len(nv)
        return self._noise_mat(nv) + self.r2 * np.ones((k, k))

    def _newton(self, nv, lam=None, beta=None, iters=8, tol=1e-3, verbose=False, max_eta_step=None):
        k = len(self.yraw)
        lam = np.zeros(k) if lam is None else lam.copy()
        x_ = self._x() if self.nuisance else np.zeros((k, 0))
        beta = np.zeros(x_.shape[1]) if beta is None else beta.copy()
        sig = self._sigma(nv)
        q0p, q0n = getattr(self, "q0p", self.abar), getattr(self, "q0n", self.bbar)
        for it in range(iters):
            xv, zv = self._mv(lam)
            qp, qn = self._q(xv, zv)
            r = self.yraw - x_ @ beta - self._gmv(qp - q0p, qn - q0n) - sig @ lam
            hm = self._hessian(qp, qn) + sig
            q = np.linalg.inv(hm)
            if x_.shape[1]:
                sm = x_.T @ q @ x_
                db = np.linalg.solve(sm, x_.T @ q @ r + x_.T @ lam)
            else:
                db = np.zeros(0)
            dl = q @ (r - x_ @ db)
            merit = float(r @ q @ r)
            t = 1.0
            for _ in range(5):  # backtrack on the Newton merit with Q frozen
                lam2, beta2 = lam + t * dl, beta + t * db
                x2, z2 = self._mv(lam2)
                qp2, qn2 = self._q(x2, z2)
                r2 = self.yraw - x_ @ beta2 - self._gmv(qp2 - q0p, qn2 - q0n) - sig @ lam2
                if float(r2 @ q @ r2) <= merit * (1 + 1e-9) or it == 0:
                    break
                t *= 0.5
            lam, beta = lam + t * dl, beta + t * db
            step = np.abs(t * dl).max() / max(np.abs(lam).max(), 1e-30)
            if verbose:
                print(f"  newton {it + 1}: step {t:g}, rel change {step:.4f}, merit {merit:.2f}", flush=True)
            if step < tol:
                break
        self.lam_exp, self.beta_exp, self.q_inv = lam, beta, q
        self.x_, self.z_ = self._mv(lam)
        self.qp, self.qn = self._q(self.x_, self.z_)
        return self

    def fit_exp(self, nu=2.0, em=3, exact=None, verbose=True, iters=8):
        """Newton on the exact dual at the current noise (from fit_noise), then Student-t E-steps refreshed on it."""
        nv = self.noise_var.copy()
        k = len(nv)
        exact = np.zeros(k, bool) if exact is None else np.asarray(exact, bool)
        base = self.noise_vector(self.k0, self.k1)
        lam, beta = None, None
        w = self.w.copy()
        for e in range(em + 1):
            self._newton(nv, lam, beta, iters=iters, verbose=verbose and e == 0)
            lam, beta = self.lam_exp, self.beta_exp
            if verbose:
                print(f"exp fit round {e}: weights < 0.5: {(w < 0.5).sum()}, sd(x - z) {np.std(self.x_ - self.z_):.3f}, "
                      f"E[n1] {self.qp.sum():.0f} E[n0] {self.qn.sum():.0f}", flush=True)
            if e == em or nu <= 0:
                break
            sn = self._noise_mat(nv)
            m = sn @ lam
            v = np.diag(sn) - np.einsum("ij,ji->i", sn @ self.q_inv, sn)
            w = (nu + 1) / (nu + (m ** 2 + np.maximum(v, 0)) / base)
            w[exact] = np.maximum(w[exact], 1.0)
            nv = self.noise_vector(self.k0, self.k1, w)
        self.w_exp, self.nv_exp = w, nv
        return self

    def score_exp(self, eta=1.0):
        return np.log(self.abar / self.bbar) + eta * (self.x_ - self.z_)

    def score_exp_one_step(self, eta=1.0):
        """Same lam as the Gaussian fit (self.lam), only the row-level map changes from linear to exponential."""
        x, z = self._mv(self.lam)
        return np.log(self.abar / self.bbar) + eta * (x - z)

    def _pairs(self, s, qp, qn):
        order = np.argsort(np.asarray(s, np.float64), kind="stable")
        b = qn[order]
        return float((qp[order] * (np.cumsum(b) - b)).sum())

    def predict_exp(self, s):
        """Posterior public score of a ranking under q (bilinear, no linearisation), relative to the base's."""
        return self.base_score + (self._pairs(s, self.qp, self.qn) - self._pairs(self.p, self.qp, self.qn)) / (self.n1 * self.n0)

    def relinearize(self, files, scores, damp=1.0):
        """Recompute g, h and the offsets around the current q (bilinear AUC model). Prior and nuisance stay."""
        qp = self.qp if damp >= 1 else self.abar + damp * (self.qp - self.abar)
        qn = self.qn if damp >= 1 else self.bbar + damp * (self.qn - self.bbar)
        wb, ub = self._wuq(self.p, qp, qn)
        pairs_b = float(qp @ wb)
        for k in range(len(self.yraw)):
            w, u = self._wuq(np.asarray(files[k], np.float64), qp, qn)
            self.g[k] = w - wb
            self.h[k] = u - ub
            self.yraw[k] = self.n1 * self.n0 * (scores[k] - self.base_score) - (float(qp @ w) - pairs_b)
        self.q0p, self.q0n = qp, qn
        return self

    def _wuq(self, s, qp, qn):
        order = np.argsort(s, kind="stable")
        q, pp = qn[order], qp[order]
        w, u = np.empty(self.n), np.empty(self.n)
        w[order] = np.cumsum(q) - q
        u[order] = pp.sum() - np.cumsum(pp)
        return w, u
