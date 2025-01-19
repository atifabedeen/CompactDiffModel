from diffusers import UNet2DModel
import torch
import torch.quantization
import torch.nn as nn
from tqdm import tqdm
import os
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from modules import UNet_conditional

# Load the UNet model used by the DDPM pipeline
model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")
device = "cpu"
# pruned_model_path = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/unet_finetuned_04.pth"
# model = torch.load(pruned_model_path, map_location=device)


# Print the architecture
#print(model)

quantize_layers = {nn.Conv2d, nn.Linear}

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    #print upto 2 decimal points
    print('Size:', os.path.getsize("temp.p")/1e6, 'MB')
    os.remove('temp.p')

print("Before quantization:")
print_size_of_model(model)

model_quantized = torch.quantization.quantize_dynamic(
    model,
    quantize_layers,
    dtype=torch.qint8
)

# for name, param in model_quantized.named_parameters():
#     print(f"Layer: {name}, Dtype: {param.dtype}")
print(model_quantized)
# model_quantized = model.half() #FP 16
# model_quantized.eval()
print("After quantization:")
print_size_of_model(model_quantized)
# ckpt = "/work/pi_pkatz_umass_edu/atif_experiments/diffusion/checkpoints/ckpt_p04_qint8.pt"
# torch.save(model_quantized, ckpt)

# # Load the entire quantized model
# device = "cpu"
# quantized_model = torch.load(ckpt, map_location=device)
# # Prepare the model for inference
# quantized_model.eval()


# device = "cuda"
# pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
# pipeline.to(device)
# pipeline.unet = quantized_model #UNCOMMENT THIS FOR QUANT

# output_dir = '/work/pi_pkatz_umass_edu/atif_experiments/diffusion/quantized_images/p04_qint8_generated_images'
# os.makedirs(output_dir, exist_ok=True)

# num_images = 2500  # Adjust the number of images as needed
# batch_size = 4
# # Loop to generate and save each image individually
# for idx in tqdm(range(0, num_images), desc="Generating and saving images"):    
#     # Generate a single image
#     with torch.no_grad():
#         with torch.cuda.amp.autocast():
#             image = pipeline(batch_size=batch_size).images[0]
    
#     # Save the image immediately
#     img_path = os.path.join(output_dir, f'image_{idx:05d}.png')  # Zero-padded numbering
#     image.save(img_path)
#     print("image saved," f'image_{idx:05d}.png')