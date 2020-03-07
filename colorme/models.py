import torch
from torch import nn
from torch.optim import Adam
import warnings
import os
import shutil
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
import contextlib
from colorme.generator import FCNGenerator
from colorme.discriminator import PatchGANDiscriminator

default_generator = FCNGenerator

default_discriminator = PatchGANDiscriminator


@contextlib.contextmanager
def TrivalContext():
    yield


class ColorMeModelMixin:
    @staticmethod
    def set_generator(generator=None, generator_kwargs=None):
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
        if generator_kwargs is None:
            generator_kwargs = dict()
        if generator is None:
            return default_generator(**generator_kwargs)
        else:
            return generator(**generator_kwargs)

    @staticmethod
    def set_discriminator(discrimnator=None, discriminator_kwargs=None):
        """

        Returns
        -------
        nn.Module
            a module that takes color images and predicts if they are real

        """
        if discriminator_kwargs is None:
            discriminator_kwargs = dict()
        if discrimnator is None:
            return default_generator(**discriminator_kwargs)
        else:
            return discrimnator(**discriminator_kwargs)

    def cuda_enumerate(self, loader):
        for iter, (X, Y) in enumerate(loader):
            if self.use_gpu:
                X = X.cuda()
                Y = Y.cuda()
            yield iter, (X, Y)

    def log_validation_images_and_loss(self, epoch, global_step,
                                       val_dataloader, writer):
        total_val_loss = 0
        total_val_samples = 0
        for j, (X_gray, X_color) in enumerate(val_dataloader):
            if self.use_gpu:
                X_gray = X_gray.cuda()
                X_color = X_color.cuda()
            output = self.forward(X_gray, train='none')
            loss = self.criterion(output, X_color)
            total_val_loss += loss.cpu().item() * len(X_gray)
            total_val_samples += len(X_gray)
            if j == 0:
                keep_gray = X_gray.cpu()
                keep_color = X_color.cpu()
                keep_colorized = output.cpu()
        # TODO the colorized may have to be inverse transformed ?
        writer.add_images('a.val_colorized', keep_colorized,
                          global_step=global_step)
        # TODO I guess if we're not shuffling we don't really
        #  need to save more than once... may want to avoid the
        #  'if' if we are going to shuffle val but I see no reason
        if global_step == 0:
            writer.add_images('b.val_grayscale_input', keep_gray,
                              global_step=global_step)
            writer.add_images('c.val_color_label', keep_color,
                              global_step=global_step)
        avg_val_loss = total_val_loss / total_val_samples
        logging.info(f"Epoch: [{epoch + 1}/"
                     f"{self.n_epochs}]\tIteration: "
                     f"[all]\tValidation "
                     f"loss: {avg_val_loss}"
                     )
        writer.add_scalar('ii.val_loss', avg_val_loss,
                          global_step=global_step)
        return avg_val_loss


class BaselineDCN(nn.Module, ColorMeModelMixin):
    def __init__(self, n_epochs: int, lr: float, logdir: str,
                 summary_interval: int = 10,
                 criterion: nn.Module = None,
                 use_gpu: bool = False,
                 validation_interval: int = 100,
                 generator: nn.Module = None,
                 generator_kwargs: dict = None,
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
        self.saved_args = locals()
        super().__init__()
        self.n_epochs = n_epochs
        self.lr = lr
        self.logdir = logdir
        self.use_gpu = use_gpu
        self.summary_interval = summary_interval
        if generator_kwargs is None:
            generator_kwargs = dict()
        self.generator_kwargs = generator_kwargs
        self.generator = self.set_generator(generator, generator_kwargs)
        if criterion is None:
            criterion = nn.MSELoss()
        self.criterion = criterion
        self.optimizer = Adam
        self.optimzer_kwargs = {'lr': self.lr}
        self.validation_interval = validation_interval
        if self.use_gpu:
            # should move the model to GPU if use_gpu is true...
            # TODO test this on GPU
            # else, we can do clf = BaselineDCN(); clf = clf.cuda(); clf.fit()
            self.__dict__.update(self.cuda().__dict__)

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
        writer = SummaryWriter(log_dir=self.logdir)
        logging.info("Starting training loop...")

        optimizer = self.optimizer(self.parameters(), **self.optimzer_kwargs)
        G_losses = []

        global_step = 0
        best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            for i, (X_gray, X_color) in enumerate(train_dataloader):
                self.zero_grad()

                if self.use_gpu:
                    X_gray = X_gray.cuda()
                    X_color = X_color.cuda()

                output = self.forward(X_gray)

                loss = self.criterion(output, X_color)
                loss.backward()
                optimizer.step()

                if i % self.summary_interval == 0:
                    # TODO add elapsed time to logs
                    logging.info(f"Epoch: [{epoch + 1}/"
                                 f"{self.n_epochs}]\tIteration: "
                                 f"[{i + 1}/{len(train_dataloader)}]\tTrain "
                                 f"loss: {loss.cpu().item()}"
                                 )
                    writer.add_scalar('i.train_loss', loss.cpu().item(),
                                      global_step=global_step)

                if (i % self.validation_interval == 0) and (
                        val_dataloader is not None):
                    val_loss = self.log_validation_images_and_loss(
                        epoch,
                        global_step,
                        val_dataloader,
                        writer)
                    is_best = False
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        is_best = True
                    checkpoint = self.prepare_checkpoint(epoch, global_step,
                                                         val_loss, optimizer)
                    self.save_checkpoint(checkpoint, is_best=is_best)
                    is_best = False


                global_step += 1

            G_losses.append(loss.cpu().item())

        writer.flush()
        writer.close()

        return G_losses

    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.logdir, f"model_{state['model_type']}__"
                                f"step_{state['global_step']}.pth"
                                )
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(
                self.logdir, 'model_best.pth'))

    def prepare_checkpoint(self, epoch, global_step, val_loss, optimizer):
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_type': self.__class__,
            'model_args': self.saved_args,
            'val_loss': val_loss,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        return checkpoint
