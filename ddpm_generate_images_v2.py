import os
from torchvision.utils import save_image
import torch
from diffusers import DDPMScheduler

# Create directories for generated samples
os.makedirs("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/original_images", exist_ok=True)
os.makedirs("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/quantized_images", exist_ok=True)


def save_generated_samples(model, noise_scheduler, num_samples, save_dir, device):
    model.eval()
    with torch.no_grad():
        # Start with random noise
        noisy_images = torch.randn((num_samples, 3, 32, 32), device=device)
        for t in reversed(range(noise_scheduler.num_train_timesteps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(noisy_images, timesteps).sample
            noisy_images = noise_scheduler.step(noisy_images, t, predicted_noise).prev_sample
        
        # Save images to the specified directory
        for i, img in enumerate(noisy_images):
            save_image((img.clamp(-1, 1) + 1) / 2,  # Rescale from [-1, 1] to [0, 1]
                       f"{save_dir}/{i}.png")


num_samples = 2000 
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# # Save samples from the original model
# save_generated_samples(unet, noise_scheduler, num_samples, "generated_samples/original", device)

import torch
from diffusers import DDPMPipeline
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
import os
# Step 1: Recreate the model architecture
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
unet = pipeline.unet

# Step 2: Set the same quantization configuration
qat_config = get_default_qat_qconfig("fbgemm")  # Use "fbgemm" or "qnnpack" as per your hardware
unet.qconfig = qat_config
unet.train()
# Step 3: Prepare the model for quantization
prepare_qat(unet, inplace=True)

# Step 4: Convert the model to a quantized version
quantized_model = convert(unet.eval(), inplace=False)

# Step 5: Load the quantized state dictionary
quantized_model.load_state_dict(torch.load("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_quantized.pth"))

# Move the model to the desired device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
quantized_model.to(device)

# Get sizes in MB
fp32_size = os.path.getsize("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_fp32.pth") / (1024 * 1024)
quantized_size = os.path.getsize("/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_quantized.pth") / (1024 * 1024)

print(f"Original Model Size (FP32): {fp32_size:.2f} MB")
print(f"Quantized Model Size (INT8): {quantized_size:.2f} MB")
print(f"Size Reduction: {100 * (fp32_size - quantized_size) / fp32_size:.2f}%")

# Save samples from the quantized model
save_generated_samples(quantized_model, noise_scheduler, num_samples, "generated_samples/quantized", device)