import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# Removed the SelfAttention class because nn.MultiheadAttention is not quantization-friendly.


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.relu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.double_conv,
            [['0', '1', '2'], ['3', '4']],
            inplace=True
        )


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    def fuse_model(self):
        self.maxpool_conv[1].fuse_model()
        self.maxpool_conv[2].fuse_model()


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # Removed SelfAttention modules
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # Removed SelfAttention
        x3 = self.down2(x2, t)
        # Removed SelfAttention
        x4 = self.down3(x3, t)
        # Removed SelfAttention

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        # Removed SelfAttention
        x = self.up2(x, x2, t)
        # Removed SelfAttention
        x = self.up3(x, x1, t)
        # Removed SelfAttention
        output = self.outc(x)
        return output

    def fuse_model(self):
        self.inc.fuse_model()
        self.down1.fuse_model()
        self.down2.fuse_model()
        self.down3.fuse_model()
        self.bot1.fuse_model()
        self.bot2.fuse_model()
        self.bot3.fuse_model()
        self.up1.fuse_model()
        self.up2.fuse_model()
        self.up3.fuse_model()


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # Removed SelfAttention modules
        self.down2 = Down(128, 256)
        # Removed SelfAttention modules
        self.down3 = Down(256, 256)
        # Removed SelfAttention modules

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # Removed SelfAttention modules
        self.up2 = Up(256, 64)
        # Removed SelfAttention modules
        self.up3 = Up(128, 64)
        # Removed SelfAttention modules
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # Removed SelfAttention
        x3 = self.down2(x2, t)
        # Removed SelfAttention
        x4 = self.down3(x3, t)
        # Removed SelfAttention

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        # Removed SelfAttention
        x = self.up2(x, x2, t)
        # Removed SelfAttention
        x = self.up3(x, x1, t)
        # Removed SelfAttention
        output = self.outc(x)
        return output

    def fuse_model(self):
        self.inc.fuse_model()
        self.down1.fuse_model()
        self.down2.fuse_model()
        self.down3.fuse_model()
        self.bot1.fuse_model()
        self.bot2.fuse_model()
        self.bot3.fuse_model()
        self.up1.fuse_model()
        self.up2.fuse_model()
        self.up3.fuse_model()


# Initialize the quantization configuration
import torch.quantization

torch.backends.quantized.engine = 'fbgemm'  # Use 'qnnpack' for ARM CPUs

# net = UNet(device="cpu")
net = UNet_conditional(num_classes=10, device="cpu")
net.eval()
# Fuse the model
net.fuse_model()

# Set the quantization configuration for QAT
net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
net.train()
# Prepare the model for QAT
torch.quantization.prepare_qat(net, inplace=True)
print(net)
print(sum([p.numel() for p in net.parameters()]))
x = torch.randn(3, 3, 64, 64)
t = x.new_tensor([500] * x.shape[0]).long()
y = x.new_tensor([1] * x.shape[0]).long()
print(net(x, t, y).shape)

# Convert the model to a quantized version after training
# net.eval()
# net_int8 = torch.quantization.convert(net)

# Test inference with the quantized model
# output = net_int8(x, t, y)
# print(output.shape)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S"
)

# Set random seeds for reproducibility
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Diffusion class (as per your original code, adjusted for CIFAR-10 images)
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                ) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x


# Function to save generated images
def save_images(images, path, **kwargs):
    import torchvision.utils as vutils
    vutils.save_image(images, path, **kwargs)


def main():
    # Hyperparameters
    epochs = 50
    batch_size = 32
    learning_rate = 1e-4
    image_size = 32  # CIFAR-10 images are 32x32
    num_classes = 10  # CIFAR-10 has 10 classes

    # Prepare CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # Use a subset for quick testing (optional)
    subset_indices = list(range(150))
    train_subset = Subset(train_dataset, subset_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    #model = UNet_conditional(c_in=3, c_out=3, num_classes=num_classes, device=device).to(device)
    #model = UNet().to(device)
    model = UNet(device=device).to(device)


    # Set up the optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Initialize the diffusion process
    diffusion = Diffusion(img_size=image_size, device=device)

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='logs')

    # Set up quantization
    torch.backends.quantized.engine = 'fbgemm'
    from torch.quantization import get_default_qat_qconfig, float_qparams_weight_only_qconfig

    model.eval()
    model.fuse_model()
    model.train()

    # Create a qconfig_dict to specify qconfig for embedding layer
    qconfig_dict = {
        '': get_default_qat_qconfig('fbgemm'),  # Default qconfig for all layers
        'label_emb': float_qparams_weight_only_qconfig,  # Specific qconfig for embedding layer
    }

    # Prepare the model for QAT with qconfig_dict
    torch.quantization.prepare_qat(model, qconfig_dict, inplace=True)

    # Training loop
    model.train()
    total_steps = 0
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        pbar = tqdm(train_dataloader)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Sample random timesteps
            t = diffusion.sample_timesteps(images.size(0)).to(device)
            # Add noise to images
            x_t, noise = diffusion.noise_images(images, t)
            x_t = x_t.to(device)
            noise = noise.to(device)
            # Predict the noise
            predicted_noise = model(x_t, t)
            # Compute loss
            loss = criterion(noise, predicted_noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            pbar.set_postfix(loss=loss.item())

            # Log to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), total_steps)
            total_steps += 1

        # Save model checkpoint every few epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            # Convert to quantized model
            model.cpu()
            quantized_model = torch.quantization.convert(model.eval(), inplace=False)
            # Save the quantized model
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/quantized_model_epoch_{epoch+1}.pt'
            torch.save(quantized_model.state_dict(), checkpoint_path)
            logging.info(f"Saved quantized model at {checkpoint_path}")
            # Move model back to device
            model.to(device)

            # Generate and save sample images
            labels_for_sampling = torch.arange(num_classes).to(device)
            sample_images = diffusion.sample(model, n=num_classes, labels=labels_for_sampling)
            os.makedirs('samples', exist_ok=True)
            save_images(
                sample_images,
                f'samples/sample_epoch_{epoch+1}.png',
                nrow=5
            )
            logging.info(f"Saved sample images at samples/sample_epoch_{epoch+1}.png")

    writer.close()
    logging.info("Training completed.")


if __name__ == "__main__":
    main()
