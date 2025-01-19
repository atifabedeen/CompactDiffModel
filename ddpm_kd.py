import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from diffusers import DDPMPipeline
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import logging
from calculate_macs import macs

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)
dataloader = trainloader

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


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(batch_size, channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
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


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, compress=1, device="cuda"):
        super().__init__()
        c_0 = 64 // compress
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, c_0)
        self.down1 = Down(c_0, c_0*2)
        self.sa1 = SelfAttention(c_0*2, 16)  # Changed from 32 to 16
        self.down2 = Down(c_0*2, c_0*4)
        self.sa2 = SelfAttention(c_0*4, 8)   # Changed from 16 to 8
        self.down3 = Down(c_0*4, c_0*4)
        self.sa3 = SelfAttention(c_0*4, 4)   # Changed from 8 to 4

        self.bot1 = DoubleConv(c_0*4, c_0*8)
        self.bot2 = DoubleConv(c_0*8, c_0*8)
        self.bot3 = DoubleConv(c_0*8, c_0*4)

        self.up1 = Up(c_0*8, c_0*2)
        self.sa4 = SelfAttention(c_0*2, 8)   # Changed from 16 to 8
        self.up2 = Up(c_0*4, c_0)
        self.sa5 = SelfAttention(c_0, 16)    # Changed from 32 to 16
        self.up3 = Up(c_0*2, c_0)
        self.sa6 = SelfAttention(c_0, 32)    # Changed from 64 to 32
        self.outc = nn.Conv2d(c_0, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, compress=1, device="cuda"):
        super().__init__()
        c_0 = 64 // compress
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, c_0)
        self.down1 = Down(c_0, c_0*2)
        self.sa1 = SelfAttention(c_0*2, 16)  # Changed from 32 to 16
        self.down2 = Down(c_0*2, c_0*4)
        self.sa2 = SelfAttention(c_0*4, 8)   # Changed from 16 to 8
        self.down3 = Down(c_0*4, c_0*4)
        self.sa3 = SelfAttention(c_0*4, 4)   # Changed from 8 to 4

        self.bot1 = DoubleConv(c_0*4, c_0*8)
        self.bot2 = DoubleConv(c_0*8, c_0*8)
        self.bot3 = DoubleConv(c_0*8, c_0*4)

        self.up1 = Up(c_0*8, c_0*2)
        self.sa4 = SelfAttention(c_0*2, 8)   # Changed from 16 to 8
        self.up2 = Up(c_0*4, c_0)
        self.sa5 = SelfAttention(c_0, 16)    # Changed from 32 to 16
        self.up3 = Up(c_0*2, c_0)
        self.sa6 = SelfAttention(c_0, 32)    # Changed from 64 to 32
        self.outc = nn.Conv2d(c_0, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            label_embedding = self.label_emb(y)
            t += label_embedding

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


from PIL import Image

def save_image(images, path, **kwargs):
    for i, image in enumerate(images):
        ndarr = image.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(path, f"{i}.jpg"))

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
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train() # why?
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train_kd(teacher, model, compress, kd_frac, lr, epochs, dataloader, run_name, ckpt):
    #setup_logging(args.run_name)
    #dataloader = get_data(args)
    # teacher = UNet(device=device).to(device)
    # teacher.load_state_dict(torch.load(args.teacher_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    teacher.eval()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    diffusion = Diffusion(img_size=32, device=device)
    #logger = SummaryWriter(os.path.join("runs", args.run_name))
    
    l = len(dataloader)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            model_predicted_noise = model(x_t, t)
            teacher_predicted_noise = teacher(x_t, t,  return_dict=False)[0] # has to have same #steps
            
            loss_base = mse(noise, model_predicted_noise)
            loss_teacher = mse(teacher_predicted_noise, model_predicted_noise)
            loss = kd_frac * loss_teacher + (1-kd_frac) * loss_base
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        #save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), ckpt)


def generate_and_save_images(model, diffusion, num_images, batch_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division
    image_count = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_images - image_count)
            # Generate images
            sampled_images = diffusion.sample(model, n=current_batch_size)
            # Save images with unique filenames
            for i, img in enumerate(sampled_images):
                filename = os.path.join(save_dir, f'image_{image_count + i:05d}.png')  # Zero-padded numbering
                ndarr = img.permute(1, 2, 0).to('cpu').numpy()
                im = Image.fromarray(ndarr)
                im.save(filename)
            image_count += current_batch_size


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    #print upto 2 decimal points
    print('Size:', os.path.getsize("temp.p")/1e6, 'MB')
    os.remove('temp.p')

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
#     pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
#     pipeline.to(device)

#     # Extract the UNet model for quantization
#     unet = pipeline.unet
#     # Set the model to train mode
#     teacher = unet
#     ckpt = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/ckpt_kd_4.pt"
#     compress = 4
#     kd_frac = 0.8
#     lr = 3e-4
#     epochs = 50
#     student = UNet(compress=compress, device=device).to(device) # model of 1/4th size
#     #train_kd(teacher, student, compress, kd_frac, lr, epochs, dataloader, "First", ckpt)

#     state_dict = torch.load(ckpt, map_location=device)
#     model = UNet(compress=compress, device=device).to(device)
#     print("generating images")
#     # Load the state dictionary into the model
#     model.load_state_dict(state_dict)
#     diffusion = Diffusion(img_size=32, device=device)
#     quantize_layers = {nn.Conv2d, nn.Linear}
#     model = torch.quantization.quantize_dynamic(
#     model,
#     quantize_layers,
#     dtype=torch.qint8
#     ) #UNCOMMENT THIS FOR QUANTIZING

#     print_size_of_model(model)
#     num_images = 2500
#     batch_size = 8
#     #generate_and_save_images(model, diffusion, num_images, batch_size, "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/kd_images/kd4_qint8")

if __name__ == "__main__":
    device = "cpu"

    model = UNet(compress=4, device=device).to(device)
    #model = torch.load("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_finetuned_01.pth")
    ckpt = torch.load("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/ckpt_kd_4.pt")
    model.load_state_dict(ckpt)
    model.to(device)
        
    # quantize_layers = {nn.Conv2d, nn.Linear}
    # model_quantized = torch.quantization.quantize_dynamic(
    # model,
    # quantize_layers,
    # dtype=torch.qint8
    # )
    macs(model)