import torch
from diffusers import DDPMPipeline
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Load the pre-trained pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipeline.to(device)

# Extract the UNet model for quantization
unet = pipeline.unet
torch.save(unet.state_dict(), "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_fp32.pth")
# Set the model to train mode
unet.train()

# Prepare the model for QAT
qat_config = get_default_qat_qconfig("fbgemm")  # Use "fbgemm" or "qnnpack" depending on your hardware
unet.qconfig = qat_config

# Prepare the model for QAT
prepare_qat(unet, inplace=True)

import torch
import torchvision
import torchvision.transforms as transforms

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


from diffusers import DDPMScheduler
import torch.nn.functional as F

# Initialize the noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Define optimizer
optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

# Set UNet to training mode
unet.train()

# Fine-tuning loop
epochs = 50  # Number of epochs
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, _ in tqdm(trainloader):
        inputs = inputs.to(device)

        # Generate random timesteps
        batch_size = inputs.size(0)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()

        # Add noise to the input images based on the timesteps
        noise = torch.randn_like(inputs)  # Random Gaussian noise
        noisy_inputs = noise_scheduler.add_noise(inputs, noise, timesteps)

        # Predict the noise using the UNet
        optimizer.zero_grad()
        predicted_noise = unet(noisy_inputs, timesteps).sample  # `.sample` gets the output tensor

        # Compute the loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader)}")

unet.eval()
unet.to("cpu")
quantized_model = torch.quantization.convert(unet, inplace=False)
torch.save(quantized_model.state_dict(), "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_quantized.pth")