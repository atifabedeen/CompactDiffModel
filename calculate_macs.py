from fvcore.nn import FlopCountAnalysis
import torch
import torch_pruning as tp

def macs(model):
    sample_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 color channels, 32x32 image
    timestep = torch.tensor([10], dtype=torch.long)  # Example timestep
    example_inputs = {'sample': sample_input, 'timestep': timestep}
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    print("#MACS: {:.4f} G".format(macs/1e9))