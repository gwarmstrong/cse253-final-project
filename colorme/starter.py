import torch
from torch import nn
from torch.optim import Adam
import warnings
import logging
from torch.utils.tensorboard import SummaryWriter
import contextlib


@contextlib.contextmanager
def TrivalContext():
    yield


class BaselineDCN(nn.Module):
    def __init__(self, n_epochs: int, lr: float, logdir: str,
                 summary_interval: int = 100,
                 criterion: nn.Module = None,
                 use_gpu: bool = False,
                 ):
        """

        Parameters
        ----------
        n_epochs : int
            how many epochs to train the model for
        lr : float
            the learning rate to use
        logdir : str
            the directory to store the files logs in
        use_gpu : bool
            Whether the model should be run on a GPU

        Examples
        --------
        >>> gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs',
        ...     criterion=nn.MSELoss(), use_gpu=False)
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.lr = lr
        self.logdir = logdir
        self.use_gpu = use_gpu
        self.summary_interval = summary_interval
        self.generator = self.set_generator()
        if criterion is None:
            criterion = nn.MSELoss()
        self.criterion = criterion
        self.optimizer = Adam
        self.optimzer_kwargs = {'lr': self.lr}
        if self.use_gpu:
            # should move the model to GPU if use_gpu is true...
            # TODO test this on GPU
            # else, we can do clf = BaselineDCN(); clf = clf.cuda(); clf.fit()
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

    def fit(self, train_dataloader, val_dataloader=None):
        """

        Parameters
        ----------
        train_dataloader : dataloader yields (X_gray, X_color)
            Gray and color image pairs of the training set.
            X_gray of shape (batch_size, N, H)
            X_color of shape (batch_size, N, H, D)
        val_dataloader : dataloader yields (X_gray, X_color)
            Gray and color image pairs of the validation set.
            X_gray of shape (batch_size, N, H)
            X_color of shape (batch_size, N, H, D)

        Returns
        -------

        """
        # writer = SummaryWriter(log_dir=self.logdir)
        logging.info("Starting training loop...")

        optimizer = self.optimizer(self.parameters(), **self.optimzer_kwargs)
        G_losses = []

        for epoch in range(self.n_epochs):
            for i, (X_gray, X_color) in enumerate(train_dataloader):
                self.zero_grad()

                if self.use_gpu:
                    X_gray = X_gray.cuda()
                    X_color = X_color.cuda()

                output = self.generator(X_gray)

                loss = self.criterion(output, X_color)
                loss.backward()
                optimizer.step()

                if i % self.summary_interval == 0:
                    logging.info(f"Epoch: [{epoch}/"
                                 f"{self.n_epochs}]\tIteration: "
                                 f"[{i}/{len(train_dataloader)}]\tTrain "
                                 f"loss: {loss.item()}"
                                 )

            G_losses.append(loss.cpu().item())

        # writer.flush()
        # writer.close()
