

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ============================================
# DEVICE CONFIGURATION
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ============================================
# DATASET CLASS
# ============================================

class DeblurDataset(Dataset):

    def __init__(self, blur_dir, sharp_dir, transform=None):

        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform

        self.blur_images = sorted(os.listdir(blur_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):

        blur_path = os.path.join(self.blur_dir, self.blur_images[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp_images[idx])

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img




transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])



dataset = DeblurDataset(
    blur_dir="dataset/blur",
    sharp_dir="dataset/sharp",
    transform=transform
)

dataloader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),

            nn.Linear(256 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)



generator = Generator().to(device)
discriminator = Discriminator().to(device)



adversarial_loss = nn.BCELoss()
pixel_loss = nn.L1Loss()


optimizer_G = optim.Adam(generator.parameters(),
                         lr=0.0002,
                         betas=(0.5, 0.999))

optimizer_D = optim.Adam(discriminator.parameters(),
                         lr=0.0002,
                         betas=(0.5, 0.999))



epochs = 20

for epoch in range(epochs):

    for i, (blur_imgs, sharp_imgs) in enumerate(dataloader):

        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)

        batch_size = blur_imgs.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)


        optimizer_G.zero_grad()

        generated_imgs = generator(blur_imgs)

        validity = discriminator(generated_imgs)

        g_adv_loss = adversarial_loss(validity, real_labels)

        g_pixel_loss = pixel_loss(generated_imgs, sharp_imgs)

        g_loss = g_adv_loss + 100 * g_pixel_loss

        g_loss.backward()

        optimizer_G.step()


        optimizer_D.zero_grad()

        real_output = discriminator(sharp_imgs)
        fake_output = discriminator(generated_imgs.detach())

        d_real_loss = adversarial_loss(real_output,
                                       real_labels)

        d_fake_loss = adversarial_loss(fake_output,
                                       fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()

        optimizer_D.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Batch [{i+1}/{len(dataloader)}] "
              f"G Loss: {g_loss.item():.4f} "
              f"D Loss: {d_loss.item():.4f}")



torch.save(generator.state_dict(),
           "generator_model.pth")

print("Generator Model Saved")

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor.to(device)

def tensor_to_image(tensor):

    image = tensor.squeeze(0).cpu().detach()

    image = image * 0.5 + 0.5

    image = image.permute(1, 2, 0).numpy()

    image = np.clip(image, 0, 1)

    return image

generator.eval()

test_image = load_image("test_blur.jpg")

with torch.no_grad():

    output = generator(test_image)

output_image = tensor_to_image(output)



plt.imshow(output_image)
plt.title("Deblurred Image")
plt.axis("off")
plt.show()

output_image = (output_image * 255).astype(np.uint8)

cv2.imwrite("deblurred_output.jpg",
            cv2.cvtColor(output_image,
                         cv2.COLOR_RGB2BGR))

print("Deblurred image saved successfully")
