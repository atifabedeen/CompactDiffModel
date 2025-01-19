import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from tqdm import tqdm
import os
from ddpm_kd import UNet, train_kd, generate_and_save_images, Diffusion
from diffusers import DDPMPipeline
from ddpm_kd import print_size_of_model


def print_size_of_model(model):
    torch.save(model, "temp.p")
    #print upto 2 decimal points
    print('Size:', os.path.getsize("temp.p")/1e6, 'MB')
    os.remove('temp.p')

# -------------------
# LoRA-enhanced UNet
# -------------------

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, factor=2, alpha=1.0):
        super().__init__()
        self.r = max(1, min(in_features, out_features) // factor)  # Compute rank dynamically
        self.alpha = alpha

        # Pre-trained weights (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # LoRA low-rank weights
        self.lora_A = nn.Parameter(torch.randn(self.r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, self.r) * 0.01)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return (
            torch.nn.functional.linear(x, self.weight, self.bias)
            + self.alpha * torch.nn.functional.linear(x, self.lora_B @ self.lora_A)
        )


class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, factor=2, alpha=1.0):
        super().__init__()
        self.in_channels = in_channels  # Store as instance attribute
        self.out_channels = out_channels  # Store as instance attribute
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.r = max(1, min(in_channels, out_channels) // factor)  # Compute rank dynamically
        self.alpha = alpha

        # Original pre-trained weights (frozen)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size), requires_grad=False
        )

        # LoRA low-rank weights
        self.lora_A = nn.Parameter(torch.randn(self.r, in_channels, 1, 1) * 0.01)  # Low-rank A
        self.lora_B = nn.Parameter(torch.randn(out_channels, self.r, 1, 1) * 0.01)  # Low-rank B

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Compute the LoRA weight update
        lora_update = torch.matmul(
            self.lora_B.flatten(1), self.lora_A.flatten(1)
        ).reshape(self.out_channels, self.in_channels, 1, 1)

        # Expand LoRA update to match kernel dimensions
        lora_update = lora_update.expand(-1, -1, *self.kernel_size)

        # Apply the LoRA update to the frozen weights
        return torch.nn.functional.conv2d(
            x, self.weight + self.alpha * lora_update, self.bias, stride=self.stride, padding=self.padding
        )



def apply_lora_to_unet(model, factor=2, alpha=1.0):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module.in_features, module.out_features, factor, alpha))
        elif isinstance(module, nn.Conv2d):
            setattr(model, name, LoRAConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding, factor=factor, alpha=alpha
            ))
        else:
            apply_lora_to_unet(module, factor, alpha)
    return model



# ------------------------
# CIFAR-10 Data Preparation
# ------------------------

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader


# ---------------------------
# DDPM Training Components
# ---------------------------

class GaussianDiffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps

    def sample_noise(self, x, t):
        noise = torch.randn_like(x)
        # Reshape and broadcast t to match x's dimensions
        t = t.view(-1, 1, 1, 1)  # Reshape t to [batch_size, 1, 1, 1]
        return (1 - t / self.timesteps) * x + (t / self.timesteps) * noise

    def loss_fn(self, x, model, t):
        noise = torch.randn_like(x)  # Generate random noise
        noisy_x = self.sample_noise(x, t)  # Add noise to the input
        predicted_output = model(noisy_x, t)  # UNet2DModel output
        predicted_noise = predicted_output.sample  # Extract the predicted noise
        return torch.nn.functional.mse_loss(predicted_noise, noise)  # Calculate MSE loss




# -------------------
# Training Loop
# -------------------

def train_lora_unet_on_cifar10(unet_model, ckpt, epochs):
    # Training setup
    train_loader = get_dataloader(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, unet_model.parameters()), lr=1e-4)
    diffusion = GaussianDiffusion(timesteps=1000)

    for epoch in range(epochs):
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            t = torch.randint(0, 1000, (x.size(0),), device=device)
            optimizer.zero_grad()

            loss = diffusion.loss_fn(x, unet_model, t)  # Corrected loss function
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    # Save the trained LoRA model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(unet_model, ckpt)

def save_lora_parameters(model):
    lora_state_dict = {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name  # Save only LoRA weights
    }
    print_size_of_model(lora_state_dict)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_ckpt_path = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/p05_lora_kd_factor_{}.pt"
    ckpt = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_finetuned_05.pth"
    # Define the factors to test
    factors = [2]  # Adjust or extend this list as needed

    # Model setup and Training
    for factor in factors:
        print(f"Testing with factor: {factor}")
        #unet_model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32") 
        unet_model = torch.load(ckpt)
        unet_model = apply_lora_to_unet(unet_model, factor=factor, alpha=1.0)

        # Freeze all non-LoRA parameters
        for param in unet_model.parameters():
            param.requires_grad = False
        for name, param in unet_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Train the model
        ckpt = base_ckpt_path.format(factor)
        epochs = 1
        train_lora_unet_on_cifar10(unet_model, ckpt, epochs)

    # GENERATE IMAGES
    for factor in factors:
        model = torch.load(base_ckpt_path.format(factor))
        output_dir = f"/work/pi_pkatz_umass_edu/atif_experiments/diffusion/kd_images/unet_p05_lora_{factor}"
        os.makedirs(output_dir, exist_ok=True)
        num_images = 2500
        batch_size = 32
        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
        pipeline.to(device)
        pipeline.unet = model
        global_idx = 0
        for batch_start in tqdm(range(0, num_images, batch_size), desc=f"Generating and saving images in batches for factor {factor}"):
            current_batch_size = min(batch_size, num_images - batch_start)  # Handle the last partial batch

            with torch.no_grad():
                # Generate a batch of images
                batch_images = pipeline(batch_size=current_batch_size).images

            # Save images in the batch
            for i, img in enumerate(batch_images):
                img_idx = batch_start + i  # Calculate global index
                img_path = os.path.join(output_dir, f'image_{img_idx:05d}.png')  # Zero-padded numbering
                img.save(img_path)

                global_idx += 1
    model = torch.load(base_ckpt_path)
    save_lora_parameters(model)


    # #KD LORA:
    # #ckpt = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/ckpt_lora_kd.pt"
    # kd_ckpt = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/ckpt_kd_4.pt"
    # state_dict = torch.load(kd_ckpt, map_location=device)
    # compress = 4
    # kd_frac = 0.8
    # lr = 2e-4
    # epochs = 30    
    # model = UNet(compress=compress, device=device).to(device)
    # model.load_state_dict(state_dict)
    # print("Model Loaded")
    # unet_model = apply_lora_to_unet(model, r=4, alpha=1.0)
    # for param in unet_model.parameters():
    #     param.requires_grad = False
    # for name, param in unet_model.named_parameters():
    #     if "lora" in name:
    #         param.requires_grad = True
    # pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    # pipeline.to(device)

    # # Extract the UNet model for quantization
    # unet = pipeline.unet
    # # Set the model to train mode
    # teacher = unet
    # dataloader = get_dataloader(batch_size=64)
    # #train_kd(teacher, unet_model, compress, kd_frac, lr, epochs, dataloader, "First", ckpt)

    # #GENERATE IMAGES
    # model_weights = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/ckpt_lora_kd_4.pt"
    # unet_model.load_state_dict(torch.load(model_weights))
    # unet_model.to(device)
    # print("LORA_KD_Model loaded")
    # diffusion = Diffusion(img_size=32, device=device)
    # num_images = 2500
    # batch_size = 8
    # generate_and_save_images(unet_model, diffusion, num_images, batch_size, "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/kd_images/kd4_lora")
    # # #PRUNED LORA
