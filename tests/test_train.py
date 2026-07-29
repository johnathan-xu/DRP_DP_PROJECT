"""Small dependency-free tests for non-private LoRA training helpers."""

from src.train import _select_samples


class FakeDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def select(self, indices):
        return FakeDataset([self.items[index] for index in indices])


def test_select_samples_limits_dataset():
    dataset = FakeDataset([1, 2, 3])
    assert _select_samples(dataset, 2).items == [1, 2]
    assert _select_samples(dataset, 0) is dataset
