import torch
from torch import nn
import warnings
import contextlib


@contextlib.contextmanager
def TrivalContext():
    yield


class BaselineDCN(nn.Module):
    def __init__(self, n_epochs: int, learning_rate: float, logdir: str,
                 summary_interval: int = 100,
                 criterion: nn.Module = None,
                 use_gpu: bool = False,
                 ):
        """

        Parameters
        ----------
        n_epochs : int
            how many epochs to train the model for
        learning_rate : float
            the learning rate to use
        logdir
        use_gpu

        Examples
        --------
        >>> gen = BaselineDCN(n_epochs=10, learning_rate=0.001, logdir='logs',
        ...     criterion=nn.MSELoss(), use_gpu=False)
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.logdir = logdir
        self.use_gpu = use_gpu
        self.summary_interval = summary_interval
        self.generator = self.set_generator()
        if criterion is None:
            criterion = nn.MSELoss()
        self.criterion = criterion
        if self.use_gpu:
            # should move the model to GPU if use_gpu is true...
            # TODO test this on GPU
            # else, we can do clf = BaselineDCN(); clf = clf.gpu(); clf.fit()
            self.__dict__.update(self.cuda().__dict__)

    def set_generator(self):
        """
        Return a network that can be used in the following way:
        gen = BaselineDCN().set_generator()
        for X, t in dataloader:
            y = self.generator(X)
            loss = criterion(y, t)

        Returns
        -------
        nn.Module
            a module that takes grayscale images and generates colored ones

        """
        raise NotImplemented()

    def forward(self, X, train='generator'):
        """
        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, H, W)
        train : choices ['generator', 'none']
            indicates which parts of the model be run with gradients

        Returns
        -------
        y: torch.tensor of shape (batch_size, H, W, N)
        Parameters

        """
        if train == 'generator':
            ctx = TrivalContext()
        elif train == 'none':
            ctx = torch.no_grad()
        else:
            raise ValueError(f"Invalid value '{train}' for argument train in "
                             f"forward.")
        with ctx:
            y = self.generator(X)

        return y
