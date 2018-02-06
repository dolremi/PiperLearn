class BaseData(object):
    def __init__(self):
        self.features = self.get_features()
        self.size = self.get_size()

    def __getitem__(self, idx):
        x, y = self.get_x(idx), self.get_y(idx)
        return x, y

    def __len__(self): return self.size

    @abstractmethod
    def get_features(self): raise NotImplementedError

    @abstractmethod
    def get_size(self): raise NotImplementedError

    @abstractmethod
    def get_x(self, i): raise NotImplementedError

    @abstractmethod
    def get_y(self, i): raise NotImplementedError

    @property
    def is_multi(self): return False

    @property
    def is_regression(self): return False
