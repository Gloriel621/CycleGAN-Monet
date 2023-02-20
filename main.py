import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import itertools

from dataset import ImageDataset
from models import Generator, Discriminator
from utils import cycle_consistency_loss, identity_loss, adversarial_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 0.0005
batch_size = 3
num_epochs = 10
lambda_cycle = 10
lambda_identity = 5

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize generator and discriminator
G_AB = Generator(input_channels=3, output_channels=3, num_residual_blocks=9).to(device)
G_BA = Generator(input_channels=3, output_channels=3, num_residual_blocks=9).to(device)
D_A = Discriminator(input_channels=3).to(device)
D_B = Discriminator(input_channels=3).to(device)

# Define loss functions
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
lambda_identity = 0.5

# Define optimizers
optimizer_G = optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# Initialize dataloader
dataset = ImageDataset("monet2photo", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Set model inputs
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        # --------------------
        # Train the generators
        # --------------------

        G_AB.zero_grad()
        G_BA.zero_grad()

        # Identity loss
        loss_id_A = identity_loss(G_BA(real_A), real_A, lambda_identity)
        loss_id_B = identity_loss(G_AB(real_B), real_B, lambda_identity)

        # Adversarial loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = adversarial_loss(D_B(fake_B), True, criterion_GAN)

        fake_A = G_BA(real_B)
        loss_GAN_BA = adversarial_loss(D_A(fake_A), True, criterion_GAN)

        # Cycle-consistency loss
        reconstructed_A = G_BA(fake_B)
        loss_cycle_A = cycle_consistency_loss(
            reconstructed_A, real_A, criterion_cycle, lambda_cycle
        )

        reconstructed_B = G_AB(fake_A)
        loss_cycle_B = cycle_consistency_loss(
            reconstructed_B, real_B, criterion_cycle, lambda_cycle
        )

        # Total generator loss
        loss_G = (
            loss_GAN_AB
            + loss_GAN_BA
            + loss_cycle_A
            + loss_cycle_B
            + loss_id_A * lambda_identity
            + loss_id_B * lambda_identity
        )

        # Backward and optimize
        loss_G.backward()
        optimizer_G.step()

        # --------------------
        # Train the discriminators
        # --------------------

        D_A.zero_grad()

        # Real loss
        loss_real = adversarial_loss(D_A(real_A), True, criterion_GAN)

        # Fake loss
        loss_fake = adversarial_loss(D_A(fake_A.detach()), False, criterion_GAN)

        # Total discriminator loss
        loss_D_A = (loss_real + loss_fake) / 2

        # Backward and optimize

        loss_D_A.backward()
        optimizer_D_A.step()

        D_B.zero_grad()

        # Real loss
        loss_real = adversarial_loss(D_B(real_B), True, criterion_GAN)

        # Fake loss
        loss_fake = adversarial_loss(D_B(fake_B.detach()), False, criterion_GAN)

        # Total discriminator loss
        loss_D_B = (loss_real + loss_fake) / 2

        # Backward and optimize
        loss_D_B.backward()
        optimizer_D_B.step()

        # Print losses
        if i % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(dataloader),
                    (loss_D_A + loss_D_B).item(),
                    loss_G.item(),
                )
            )

        # Save generated images
        if i % 100 == 0:
            save_image(fake_A, "generated_A_%d.png" % (epoch * len(dataloader) + i))
            save_image(fake_B, "generated_B_%d.png" % (epoch * len(dataloader) + i))


        
