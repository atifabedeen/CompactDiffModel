# CompactDiffModel

**For more information and details regarding experimentation design and results, please check out the paper (PDF is in the GitHub repository).**

CompactDiffModel is an implementation of a compact diffusion model employing various compression techniques, including pruning, quantization, low-rank adaptation, and knowledge distillation. The model is trained and fine-tuned on the CIFAR-10 dataset to achieve efficient performance without significant loss in image generation quality.

## Overview

Diffusion models have demonstrated remarkable capabilities in generating high-quality images. However, their substantial computational and memory demands pose challenges for deployment in resource-constrained environments. CompactDiffModel addresses these challenges by integrating multiple model compression strategies:

- **Pruning**: Removing redundant parameters to reduce model size and complexity.
- **Quantization**: Reducing the precision of model parameters to lower memory usage and computational requirements.
- **Low-Rank Adaptation (LoRA)**: Decomposing weight matrices into low-rank representations to achieve efficient fine-tuning.
- **Knowledge Distillation**: Transferring knowledge from a larger teacher model to a smaller student model to maintain performance while reducing size.

By combining these techniques, CompactDiffModel achieves a balance between model compactness and performance, making it suitable for applications with limited computational resources.

## Repository Structure

The repository includes the following key components:

- **`ddpm_quant.py`**: Implementation of quantization-aware training for the diffusion model.
- **`ddpm_prune.py`**: Scripts for applying pruning techniques to the diffusion model.
- **`ddpm_kd.py`**: Knowledge distillation implementation for transferring knowledge from a teacher to a student model.
- **`lora_v2.py`**: Implementation of Low-Rank Adaptation for efficient fine-tuning.
- **`calculate_fid_v1.py`**: Script to calculate the Fréchet Inception Distance (FID) for evaluating image generation quality.
- **`calculate_macs.py`**: Utility to compute Multiply-Accumulate Operations (MACs) for assessing computational complexity.
- **`LiteDiffusion Report.pdf`**: Comprehensive report detailing the methodologies, experiments, and results of the CompactDiffModel project.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/atifabedeen/CompactDiffModel.git
   ```

2. Navigate to the project directory:

   ```bash
   cd CompactDiffModel
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Training and Fine-Tuning

The repository provides scripts for training and fine-tuning the diffusion model using the aforementioned compression techniques. Detailed instructions and configurations are available in the `LiteDiffusion Report.pdf`. It is recommended to review this report to understand the experimental setup and reproduce the results.

## Evaluation

To assess the performance of the compressed models, use the `calculate_fid_v1.py` script to compute the Fréchet Inception Distance (FID), which measures the quality of generated images. Additionally, the `calculate_macs.py` script helps evaluate the computational efficiency of the models by calculating the number of Multiply-Accumulate Operations (MACs).

## Results

The application of pruning, quantization, low-rank adaptation, and knowledge distillation has led to a significant reduction in model size and computational complexity. Notably, these compression techniques have been implemented in various diffusion model projects, demonstrating their effectiveness in reducing model size while maintaining performance.

For a comprehensive analysis of the results, including quantitative metrics and qualitative assessments, please refer to the `LiteDiffusion Report.pdf` included in this repository.

## Acknowledgements

This project builds upon foundational work in diffusion models and model compression techniques. We acknowledge the contributions of the research community in these areas, which have been instrumental in the development of CompactDiffModel.

## Contact

For any questions or inquiries, please contact [Atif Abedeen](mailto:atifabedeen@example.com).

---

By integrating multiple compression strategies, CompactDiffModel serves as a step forward in making diffusion models more accessible and practical for real-world applications, especially in environments with limited computational resources.
