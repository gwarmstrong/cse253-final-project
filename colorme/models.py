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

    def __init__(self, *args, **kwargs):
        raise NotImplemented()

    @staticmethod
    def set_generator(generator=None, generator_kwargs=None) -> nn.Module:
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
    def set_discriminator(discrimnator=None, discriminator_kwargs=None) -> \
            nn.Module:
        """

        Returns
        -------
        nn.Module
            a module that takes color images and predicts if they are real

        """
        if discriminator_kwargs is None:
            discriminator_kwargs = dict()
        if discrimnator is None:
            return default_discriminator(**discriminator_kwargs)
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
            with torch.no_grad():
                output = self.generator(X_gray)
            loss = self.Gcriterion(output, X_color)
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

    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.logdir, f"model_{state['model_type']}__"
                                             f"step_{state['global_step']}.pth"
                                )
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(
                self.logdir, 'model_best.pth'))

    def prepare_checkpoint(self, *args, **kwargs):
        raise NotImplemented()


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
        self.Gcriterion = criterion
        self.Goptimizer = Adam
        self.Goptimzer_kwargs = {'lr': self.lr}
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

        optimizer = self.Goptimizer(self.parameters(), **self.Goptimzer_kwargs)
        G_losses = []

        global_step = 0
        best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            for i, (X_gray, X_color) in enumerate(train_dataloader):
                self.zero_grad()

                if self.use_gpu:
                    X_gray = X_gray.cuda()
                    X_color = X_color.cuda()

                output = self(X_gray)

                loss = self.Gcriterion(output, X_color)
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


class BaselineDCGAN(nn.Module, ColorMeModelMixin):

    def __init__(self, n_epochs: int, lr: float, logdir: str,
                 summary_interval: int = 10,
                 generator_criterion: nn.Module = None,
                 discriminator_criterion: nn.Module = None,
                 use_gpu: bool = False,
                 validation_interval: int = 100,
                 generator: nn.Module = None,
                 generator_kwargs: dict = None,
                 discriminator: nn.Module = None,
                 discriminator_kwargs: dict = None,
                 training_loop: str = 'auto',
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
        if discriminator_kwargs is None:
            discriminator_kwargs = dict()
        self.generator_kwargs = generator_kwargs
        self.generator = self.set_generator(generator, generator_kwargs)
        self.discriminator = self.set_discriminator(discriminator,
                                                    discriminator_kwargs)
        if generator_criterion is None:
            generator_criterion = nn.MSELoss()
        if discriminator_criterion is None:
            discriminator_criterion = nn.BCELoss()
        self.Gcriterion = generator_criterion
        self.Dcriterion = discriminator_criterion
        self.Goptimizer = Adam
        self.Doptimizer = Adam
        self.Goptimzer_kwargs = {'lr': self.lr}
        self.Doptimzer_kwargs = {'lr': self.lr}
        self.validation_interval = validation_interval
        self.training_loop = training_loop
        self.real_label = 0
        self.fake_label = 1
        if self.use_gpu:
            # should move the model to GPU if use_gpu is true...
            # TODO test this on GPU
            # else, we can do clf = BaselineDCN(); clf = clf.cuda(); clf.fit()
            self.__dict__.update(self.cuda().__dict__)

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
        training_loops = {
            # add new training loops here
            'baseline': self._baseline_loop,
            'auto': self._baseline_loop,
        }
        training_loop = training_loops[self.training_loop]
        logging.info("Starting training loop...")

        Goptimizer = self.Goptimizer(self.generator.parameters(),
                                     **self.Goptimzer_kwargs)
        Doptimizer = self.Doptimizer(self.discriminator.parameters(),
                                     **self.Doptimzer_kwargs)
        G_losses = []
        D_losses = []

        # TODO adapt training loop to GAN... from tutorial roughly
        global_step = 0
        best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            Gloss, Dloss, global_step = training_loop(Goptimizer,
                                                      Doptimizer,
                                                      best_val_loss,
                                                      epoch,
                                                      global_step,
                                                      train_dataloader,
                                                      val_dataloader,
                                                      writer
                                                      )

            G_losses.append(Gloss)
            D_losses.append(Dloss)

        writer.flush()
        writer.close()

        return G_losses, D_losses

    def _baseline_loop(self,
                       Goptimizer,
                       Doptimizer,
                       best_val_loss,
                       epoch,
                       global_step,
                       train_dataloader,
                       val_dataloader,
                       writer
                       ):
        for i, (X_gray, X_color) in enumerate(train_dataloader):
            self.zero_grad()

            if self.use_gpu:
                X_gray = X_gray.cuda()
                X_color = X_color.cuda()

            # this is COMPLETELY copy paste and edit from pytorch
            # tutorials
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            self.discriminator.zero_grad()
            # Format batch
            b_size = X_color.size(0)
            label = torch.full((b_size,), self.real_label)
            if self.use_gpu:
                label = label.cuda()
            # Forward pass real batch through D
            output = self.discriminator(X_color).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.Dcriterion(output, label)
            errD_real.backward()

            # Train with all-fake batch
            # Generate fake image batch with G
            fake = self.generator(X_gray)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.Dcriterion(output, label)
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Calculate gradients for D in backward pass
            # errD.backward()

            # Update D
            Doptimizer.step()

            ############################
            # (2) Update G network: minimize (1 - log(D(G(z)))) + || t - G(z) ||
            #  where z is grayscale and t is color of the same image
            ###########################
            self.generator.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator
            errG_color = self.Gcriterion(fake, X_color)
            # cost
            # Since we just updated D, perform another forward pass of
            # all-fake batch through D
            output = self.discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG_fool = self.Gcriterion(output, label)
            # Calculate gradients for G
            errG = errG_fool + errG_color
            errG.backward()

            # Update G
            Goptimizer.step()

            ############################
            # END OF PYTORCH TUTORIAL LOOP
            ############################

            Gloss = errG.cpu().item()
            Dloss = errD.cpu().item()

            # TODO log the losses
            if i % self.summary_interval == 0:
                # TODO add elapsed time to logs
                logging.info(f"Epoch: [{epoch + 1}/"
                             f"{self.n_epochs}]\tIteration: "
                             f"[{i + 1}/{len(train_dataloader)}]\tTrain "
                             f"Gloss: {Gloss}\tTrain Dloss: {Dloss}"
                             )
                writer.add_scalar('i.train_loss/generator', Gloss,
                                  global_step=global_step)
                writer.add_scalar('i.train_loss/discriminator', Dloss,
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
                                                     val_loss, Goptimizer,
                                                     Doptimizer,
                                                     )
                self.save_checkpoint(checkpoint, is_best=is_best)
                is_best = False

            global_step += 1
        return Gloss, Dloss, global_step

    def forward(self, X, train='none', skip_generator=False):
        """
        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, H, W)
        train : choices ['discriminator', 'both', 'generator', 'none']
            indicates which parts of the model be run with gradients
        skip_generator : bool
            if True, the input is fed into the generator. If False,
            it is fed directly to the discriminator

        Returns
        -------
        col_im : torch.tensor of shape (batch_size, H, W, N)
            colored image predicted from X
        disc : torch.tensor of shape (batch_size, 1)
            probability that the image is real (1 for real, 0 for fake).
            can check self.real_label and self.fake_label

        """
        if train in {'generator', 'both'}:
            genctx = TrivalContext()
        else:
            genctx = torch.no_grad()
        if train in {'discriminator', 'both'}:
            disctx = TrivalContext()
        else:
            disctx = torch.no_grad()
        if train not in {'discriminator', 'both', 'generator', 'none'}:
            raise ValueError(f"Invalid value '{train}' for argument train in "
                             f"forward.")
        if not skip_generator:
            with genctx:
                im_col = self.generator(X)
        else:
            im_col = X

        with disctx:
            disc = self.discriminator(im_col)

        return im_col, disc

    def prepare_checkpoint(self, epoch, global_step, val_loss, Goptimizer,
                           Doptimizer):
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_type': self.__class__,
            'model_args': self.saved_args,
            'val_loss': val_loss,
            'state_dict': self.state_dict(),
            'Gstate_dict': self.generator.state_dict(),
            'Dstate_dict': self.discriminator.state_dict(),
            'Goptimizer': Goptimizer.state_dict(),
            'Doptimizer': Doptimizer.state_dict(),
        }
        return checkpoint
