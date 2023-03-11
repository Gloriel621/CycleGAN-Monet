import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import itertools

from models import Generator, Discriminator
from utils import cycle_consistency_loss, identity_loss, adversarial_loss

class CycleGAN_Trainer:
    def __init__(self, lr, batch_size, num_epochs, lambda_cycle, lambda_identity, train_dataset, device):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.train_dataset = train_dataset
        self.device = device
        self.img_save_dir_A = "images/monet"
        self.img_save_dir_B = "images/new_picture"
        self.save_path = "models"

        self.best_loss = float('inf')
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.G_AB = Generator(input_channels=3, output_channels=3, num_residual_blocks=9).to(device)
        self.G_BA = Generator(input_channels=3, output_channels=3, num_residual_blocks=9).to(device)
        self.D_A = Discriminator(input_channels=3).to(device)
        self.D_B = Discriminator(input_channels=3).to(device)
        
        self.criterion_GAN = nn.MSELoss().to(device)
        self.criterion_cycle = nn.L1Loss().to(device)
        
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        self.dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        for path in [self.img_save_dir_A, self.img_save_dir_B, self.save_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        
    def train(self):
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(self.dataloader):
                # Set model inputs
                real_A = batch["A"].to(self.device)
                real_B = batch["B"].to(self.device)

                # --------------------
                # Train the generators
                # --------------------

                self.G_AB.zero_grad()
                self.G_BA.zero_grad()

                # Identity loss
                loss_id_A = identity_loss(self.G_BA(real_A), real_A, self.lambda_identity)
                loss_id_B = identity_loss(self.G_AB(real_B), real_B, self.lambda_identity)

                # Adversarial loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = adversarial_loss(self.D_B(fake_B), True, self.criterion_GAN)

                fake_A = self.G_BA(real_B)
                loss_GAN_BA = adversarial_loss(self.D_A(fake_A), True, self.criterion_GAN)

                # Cycle-consistency loss
                reconstructed_A = self.G_BA(fake_B)
                loss_cycle_A = cycle_consistency_loss(
                    reconstructed_A, real_A, self.criterion_cycle, self.lambda_cycle
                )

                reconstructed_B = self.G_AB(fake_A)
                loss_cycle_B = cycle_consistency_loss(
                    reconstructed_B, real_B, self.criterion_cycle, self.lambda_cycle
                )

                # Total generator loss
                loss_G = (
                    loss_GAN_AB
                    + loss_GAN_BA
                    + loss_cycle_A
                    + loss_cycle_B
                    + loss_id_A * self.lambda_identity
                    + loss_id_B * self.lambda_identity
                )

                # Backward and optimize
                loss_G.backward()
                self.optimizer_G.step()

                # --------------------
                # Train the discriminators
                # --------------------

                self.D_A.zero_grad()

                # Real loss
                loss_real = adversarial_loss(self.D_A(real_A), True, self.criterion_GAN)

                # Fake loss
                loss_fake = adversarial_loss(self.D_A(fake_A.detach()), False, self.criterion_GAN)

                # Total discriminator loss
                loss_D_A = (loss_real + loss_fake) / 2

                # Backward and optimize

                loss_D_A.backward()
                self.optimizer_D_A.step()

                self.D_B.zero_grad()

                # Real loss
                loss_real = adversarial_loss(self.D_B(real_B), True, self.criterion_GAN)

                # Fake loss
                loss_fake = adversarial_loss(self.D_B(fake_B.detach()), False, self.criterion_GAN)

                # Total discriminator loss
                loss_D_B = (loss_real + loss_fake) / 2

                # Backward and optimize
                loss_D_B.backward()
                self.optimizer_D_B.step()

                # Print losses
                if i % 100 == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %(epoch, self.num_epochs, i,
                            len(self.dataloader), (loss_D_A + loss_D_B).item(), loss_G.item(),
                        )
                    )

                # Save generated images
                if i % 250 == 0:
                    pathA = os.path.join(self.img_save_dir_A,"generated_A_%d.png" % (epoch * len(self.dataloader) + i))
                    pathB = os.path.join(self.img_save_dir_B,"generated_B_%d.png" % (epoch * len(self.dataloader) + i))
                    save_image(fake_A, pathA)
                    save_image(fake_B, pathB)

                if (loss_G) < self.best_loss:
                    print("Generator loss decreased from {:.4f} to {:.4f}. Saving model...".format(self.best_loss, (loss_G)))
                    torch.save(self.G_AB.state_dict(), self.save_path + '/cyclegan-best-weight-fake-monet.pth')
                    torch.save(self.G_BA.state_dict(), self.save_path + '/cyclegan-best-weight-real-picture.pth')
                    self.best_loss = loss_G