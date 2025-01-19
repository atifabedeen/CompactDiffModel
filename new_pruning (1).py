# import diffusers.models.unets.unet_2d_blocks as models
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
from diffusers import (DDIMPipeline, DDIMScheduler, DDPMPipeline,
                       DDPMScheduler, UNet2DModel)
from diffusers.models.unets.unet_2d_blocks import (DownBlock2D, UNetMidBlock2D,
                                                   UpBlock2D)

# print(dir(models))
## Load DDPM pipeline
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
# pipeline = DDPMPipeline.from_pretrained("./ddpm_cifar10_32")
scheduler = pipeline.scheduler

# Load the pre-trained UNet2DModel
model = pipeline.unet.eval()
print(model)
# Create example inputs
sample_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 color channels, 32x32 image
timestep = torch.tensor([10], dtype=torch.long)  # Example timestep
example_inputs = {'sample': sample_input, 'timestep': timestep}

# Define the importance criterion (L1 Norm)
importance = tp.importance.MagnitudeImportance(p=2)  # L1 Norm

# Identify layers to ignore
# ignored_layers = [model.conv_out, model.time_embedding, 
#                   model.down_blocks[1].attentions, model.up_blocks[2].attentions,
#                   model.mid_block.attentions]  # Exclude time embedding and output conv
ignored_layers = [model.conv_out]  # Exclude time embedding and output conv
# for name, module in model.named_modules():
#     if 'time_emb_proj' in name:
#         ignored_layers.append(module)
# Additionally, exclude all time_emb_proj layers
for down_block in model.down_blocks:
    for resnet in down_block.resnets:
        if hasattr(resnet, 'time_emb_proj'):
            ignored_layers.append(resnet.time_emb_proj)

for up_block in model.up_blocks:
    for resnet in up_block.resnets:
        if hasattr(resnet, 'time_emb_proj'):
            ignored_layers.append(resnet.time_emb_proj)
# Exclude specific layers from pruning
p = 0.4
p_string = str(p).replace(".", "_")

# Create the pruner
pruner = tp.pruner.MagnitudePruner(
    model=model,
    example_inputs=example_inputs,
    importance=importance,
    global_pruning=False,
    # pruning_ratio=p,  # Prune 20% of the channels
    ch_sparsity=p,
    iterative_steps=1,
    ignored_layers=ignored_layers,
    channel_groups={},
)


# # Define importance criteria
# conv_importance = tp.importance.MagnitudeImportance()  # L1 Norm for convs
# groupnorm_importance = tp.importance.GroupNormImportance()  # GroupNorm Importance

# Identify layers to ignore
# ignored_layers = [model.time_proj, model.conv_out, model.conv_norm_out]  # Exclude time embedding and output conv

# Apply magnitude pruning to convolutional layers and group importance to GroupNorm layers
# pruning_plan = []

# pruning_plan.append(
#     tp.pruner.MagnitudePruner(
#         model=model,
#         example_inputs=example_inputs,
#         importance=conv_importance,
#         global_pruning=False,
#         pruning_ratio=p,  # Prune 40% of channels in convs
#         ignored_layers=ignored_layers,
        
#     )
# )
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
model.zero_grad()
model.eval()

# Apply pruning
# for g in pruner.step(interactive=True):
#     print(g)
#     g.prune()
pruner.step()

from diffusers.models.resnet import Downsample2D, Upsample2D

# Update static attributes (if necessary)
for m in model.modules():
    if isinstance(m, (Upsample2D, Downsample2D)):
        m.channels = m.conv.in_channels
        m.out_channels = m.conv.out_channels  # Corrected typo here

macs, params = tp.utils.count_ops_and_params(model, example_inputs)
print(model)
print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
model.zero_grad()
del pruner


save_path = os.path.join(f"./pruned_model_timeproj", p_string)
os.makedirs(save_path, exist_ok=True)
pipeline.unet = model
pipeline.scheduler = scheduler
pipeline.save_pretrained(save_path)


os.makedirs(save_path, exist_ok=True)
torch.save(model, os.path.join(save_path, "unet_pruned.pth"))

# Sampling images from the pruned model
model = torch.load(os.path.join(save_path, "unet_pruned.pth"))
pipeline = DDPMPipeline(unet=model, scheduler=DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").scheduler)

with torch.no_grad():
    generator = torch.Generator(device=pipeline.device).manual_seed(25)
    pipeline.to("cpu")
    images = pipeline(num_inference_steps=1000, batch_size=1, generator=generator, output_type="numpy").images
    os.makedirs(os.path.join(save_path, 'vis'), exist_ok=True)
    torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(save_path))