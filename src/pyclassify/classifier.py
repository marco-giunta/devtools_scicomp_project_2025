from .utils import distance, majority_vote

class kNN:
    def __init__(self, k: int):
        if not isinstance(k, int):
            raise TypeError("k is not an int")
        if k < 1:
            raise ValueError("k is not larger than or equal to 1")
        self.k = k
    def _get_k_nearest_neighbors(self, X: list[list[float]], y: list[int], x: list[float]):
        distances = []
        for sample in X:
            d = distance(sample, x)
            distances.append(d)
        idx_sorted = sorted(range(len(distances)), key = distances.__getitem__)
        return [y[i] for i in idx_sorted[:self.k]]
    def __call__(self, data: tuple, new_points: list[list[float]]):
        X, y = data
        predictions = []
        for point in new_points:
            labels_k_nn = self._get_k_nearest_neighbors(X, y, point)
            predictions.append(majority_vote(labels_k_nn))
        return predictions