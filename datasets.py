import numpy as np


class Dataset:
    def __init__(self, train: bool = True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, item):
        assert np.isscalar(item)
        if self.label is None:
            return self.transform(self.data[item]), None
        else:
            return self.transform(self.data[item]), self.target_transform(self.label[item])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        raise NotImplementedError()
