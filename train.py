import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
DATA_DIR = Path(r"C:\Users\karth\Downloads\deblur\data")
BATCH_SIZE = 4
EPOCHS = 20
LR = 2e-4
IMG_SIZE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# DATASET
# ==========================
class DeblurDataset(Dataset):
    def __init__(self, root_dir):
        self.blur_dir = root_dir / "blurred"
        self.sharp_dir = root_dir / "sharp"

        self.images = sorted(os.listdir(self.blur_dir))

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        blur = Image.open(self.blur_dir / img_name).convert("RGB")
        sharp = Image.open(self.sharp_dir / img_name).convert("RGB")

        return self.transform(blur), self.transform(sharp)

# ==========================
# GENERATOR (U-Net Lite)
# ==========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())

        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        u1 = self.up1(d3)
        u2 = self.up2(u1)
        out = self.up3(u2)

        return out

# ==========================
# DISCRIMINATOR (PatchGAN)
# ==========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, blur, sharp):
        x = torch.cat([blur, sharp], dim=1)
        return self.model(x)

# ==========================
# TRAIN
# ==========================
def train():
    dataset = DeblurDataset(DATA_DIR / "train")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    BCE = nn.BCELoss()
    L1 = nn.L1Loss()

    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for blur, sharp in loop:
            blur, sharp = blur.to(DEVICE), sharp.to(DEVICE)

            # ------------------
            # Train Discriminator
            # ------------------
            fake = G(blur)

            real_pred = D(blur, sharp)
            fake_pred = D(blur, fake.detach())

            loss_D = (BCE(real_pred, torch.ones_like(real_pred)) +
                      BCE(fake_pred, torch.zeros_like(fake_pred))) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ------------------
            # Train Generator
            # ------------------
            fake_pred = D(blur, fake)

            loss_G = BCE(fake_pred, torch.ones_like(fake_pred)) + 100 * L1(fake, sharp)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())

    torch.save(G.state_dict(), "generator.pth")
    print("✅ Generator saved!")

# ==========================
if __name__ == "__main__":
    train()



                
