import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metric import score


def test_hand_computed():
    # positives {0.8, 0.4}, negatives {0.5, 0.1}: pairs (0.8>0.5, 0.8>0.1, 0.4<0.5, 0.4>0.1) -> 3/4
    assert abs(score([1, 0, 1, 0], [0.8, 0.5, 0.4, 0.1]) - 0.75) < 1e-12


def test_bool_labels_and_ties():
    # one tied pair counts 1/2: positives {0.5}, negatives {0.5, 0.2} -> (0.5 + 1) / 2
    assert abs(score([True, False, False], [0.5, 0.5, 0.2]) - 0.75) < 1e-12
