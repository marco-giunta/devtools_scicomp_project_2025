from pyclassify.utils import distance, majority_vote
from pyclassify.classifier import kNN
import pytest

def test_distance():
    assert distance([1, 2, 3], [2, 3, 4]) == 3
def test_majority_vote():
    assert majority_vote([1, 0, 0, 0]) == 0
def test_kNN_constructor():
    with pytest.raises(TypeError):
        kNN(2.3)
        kNN('2')
        kNN([1])
    with pytest.raises(ValueError):
        kNN(0)