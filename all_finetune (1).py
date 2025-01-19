import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
from torch.utils.data import DataLoader, Subset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transformations (normalize images between -1 and 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load the CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Randomly select 10,000 indices
# total_images = len(cifar10_dataset)
# subset_size = 100
# subset_indices = np.random.choice(total_images, subset_size, replace=False)
# subset_indices = subset_indices.tolist()

# # Create the subset
# cifar10_subset = Subset(cifar10_dataset, subset_indices)

# Create a DataLoader
batch_size = 64  # Adjust based on your GPU memory
dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)

# Number of epochs
num_epochs = 25  # Adjust as needed

# Path to the directory containing the pruned models
pruned_models_dir = "./pruned_model_timeproj"

# Get the list of subdirectories
subdirs = [d for d in os.listdir(pruned_models_dir) if os.path.isdir(os.path.join(pruned_models_dir, d))]

for subdir in subdirs:
    # For each subdirectory (e.g., "0_1", "0_2", ...)
    subdir_path = os.path.join(pruned_models_dir, subdir)
    print(f"\nProcessing pruned model in {subdir_path}")

    # Load the pruned model using torch.load
    pruned_model_path = os.path.join(subdir_path, 'unet_pruned.pth')
    if os.path.exists(pruned_model_path):
        try:
            # Load the model weights
            model = torch.load(pruned_model_path, map_location=device)
            print(f"Loaded model from {pruned_model_path}")

        except Exception as e:
            print(f"Error loading model from {pruned_model_path}: {e}")
            continue
    else:
        print(f"Model file {pruned_model_path} not found.")
        continue

    model.to(device)
    model.train()

    # Load the scheduler
    scheduler_path = os.path.join(subdir_path, "scheduler")
    if os.path.exists(scheduler_path):
        noise_scheduler = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").scheduler
        print("Loaded default scheduler from google/ddpm-cifar10-32")
        

    # Create pipeline
    pipeline = DDPMPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )
    pipeline.to(device)

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-9)

    # Finetuning loop
    for epoch in range(num_epochs):
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        for batch in progress_bar:
            images, _ = batch
            images = images.to(device)

            # Sample random timesteps for each image
            batch_size = images.size(0)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

            # Add noise to the images
            noise = torch.randn_like(images).to(device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Forward pass
            optimizer.zero_grad()
            noise_pred = model(noisy_images, timesteps).sample

            # Compute loss
            loss = criterion(noise_pred, noise)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Update progress bar
            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        lr_scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")

    # Save the fine-tuned model and pipeline
    finetuned_model_path = os.path.join(subdir_path, "finetuned_model")
    os.makedirs(finetuned_model_path, exist_ok=True)

    # Save the model weights
    torch.save(model, os.path.join(finetuned_model_path, "unet_finetuned.pth"))

    # Save the model's configuration
    # model.save_pretrained(finetuned_model_path)

    # Update the pipeline with the fine-tuned model and scheduler
    pipeline.unet = model
    pipeline.scheduler = noise_scheduler

    # Save the pipeline
    pipeline.save_pretrained(finetuned_model_path)
    print(f"Fine-tuned model and pipeline saved to {finetuned_model_path}")
