import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from random import sample

# Step 1: Load and Transform Real CIFAR-10 Images
def load_real_images(batch_size=64, limit=2500):
    # Define transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize((299, 299)),            # Resize to Inception's input size
        transforms.ToTensor(),                   # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5),    # Normalize to [-1, 1]
                             (0.5, 0.5, 0.5))
    ])
    # Load CIFAR-10 test set
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Limit to 2,500 images
    indices = list(range(len(cifar10_test)))
    limited_indices = sample(indices, limit)  # Randomly select 2,500 indices
    limited_dataset = Subset(cifar10_test, limited_indices)
    
    # Create DataLoader
    test_loader = DataLoader(limited_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Step 2: Load Generated Images and Apply Transformations
def load_generated_images(generated_images_dir, limit=2500):
    # Define transformations for generated images (same as real images)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load generated images from directory
    generated_images = []
    for img_file in os.listdir(generated_images_dir):
        img_path = os.path.join(generated_images_dir, img_file)
        img = Image.open(img_path).convert('RGB')  # Open image and ensure 3 channels
        img = transform(img)                      # Apply transformations
        generated_images.append(img)
    
    # Limit to 2,500 images
    generated_images = generated_images[:limit]
    return torch.stack(generated_images)  # Stack into a single tensor

# Step 3: Calculate FID Score
def calculate_fid(real_images_loader, generated_images_tensor, device='cuda'):
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    # Add real images to FID metric
    for real_images, _ in real_images_loader:
        fid.update(real_images.to(device), real=True)
    
    # Add generated images to FID metric
    fid.update(generated_images_tensor.to(device), real=False)
    
    # Compute and return FID score
    return fid.compute()

# Main Function
if __name__ == "__main__":
    # Set paths and device
    generated_images_dir = '/work/pi_pkatz_umass_edu/atif_experiments/diffusion/kd_images/kd_2'  # Path to your generated images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real and generated images
    print("Loading real CIFAR-10 images...")
    real_images_loader = load_real_images(limit=2500)

    print("Loading generated images...")
    generated_images_tensor = load_generated_images(generated_images_dir, limit=2500)

    # Calculate FID score
    print("Calculating FID score...")
    fid_score = calculate_fid(real_images_loader, generated_images_tensor, device=device)

    print(f"FID Score: {fid_score}")
