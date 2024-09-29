# Single Image 3D Reconstruction Using NeRF

## Introduction
This project implements a method for high-fidelity 3D reconstruction from a single 2D image by using NeRF and diffusion models. The key objective is to improve upon traditional 3D reconstruction methods by incorporating diffusion priors for enhanced texture and geometric details.

## Features
- **Two-Stage Optimization Process**: Implements a novel two-stage approach. The first stage focuses on generating an initial 3D geometric structure using NeRF with positional encoding, hierarchical sampling, and volumetric rendering techniques. The second stage refines the model by converting the rough geometry into a textured point cloud, enhancing visual details using high-quality texture mapping and diffusion priors.
  
- **High-Fidelity 3D Reconstruction**: Leveraged NeRF’s ability to simulate complex lighting effects to achieves highly detailed and realistic 3D models from a single 2D image.

- **Diffusion-Based Texture Enhancement**: Enhances model texture and detail quality through the integration of diffusion priors during the refinement stage, achieving a more realistic and visually accurate 3D model.

- **Broad Application Potential**: Applicable to real-world 3D scene modeling, cultural heritage preservation, and high-quality AIGC generation.

## Technologies Used
- **NeRF**: Utilized Mip-NeRF for efficient 3D scene representation and multi-scale neural rendering, improving geometric detail and scene reconstruction quality.
  
- **Diffusion Model**: Employed Stable Diffusion 2.0 to enhance texture quality and generate high-fidelity visual details during the refinement stage.

- **SAM**: Applied SAM to segment foreground masks from input images, ensuring accurate 3D reconstruction by isolating relevant visual elements.

- **BLIP-2**: Used BLIP-2 to generate reliable textual descriptions from input images, aiding in semantic understanding and model optimization.

- **DPT**: Incorporated DPT for depth estimation of foreground images, facilitating accurate geometric modeling in 3D reconstruction.

- **Tiny-CUDA-NN**: Leveraged Tiny-CUDA-NN as the NeRF training framework, enabling faster training and inference by optimizing GPU usage.

- **PyTorch3D**: Utilized PyTorch3D to handle complex 3D data processing, including mesh and point cloud manipulations for efficient model creation.

- **Contextual Loss**: Implemented contextual loss to evaluate and improve the quality of 3D model reconstructions, ensuring visual fidelity and accuracy.

- **Raymarching**: Enabled fast real-time scene rendering through raymarching, providing efficient visualization of complex 3D environments.


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/single-image-3d-reconstruction.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download [DPT weights](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt) and put it under \DPT\weights.
4. Download [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) and put it under \openai.
5. Download [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put it under \segment-anything-main\models.
6. Download [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2) and put it under \stabilityai.
7. Download [BLIP-2](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth) and put it under \salesforce.
