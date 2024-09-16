# Stable Diffusion PyTorch Implementation

![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains a PyTorch implementation of the Stable Diffusion model for text-to-image and image-to-image generation tasks. It provides a flexible and customizable framework for generating high-quality images based on text prompts or existing images.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- 🖼️ Text-to-image generation
- 🧠 Classifier-free guidance (CFG) support
- 🔢 DDPM (Denoising Diffusion Probabilistic Models) sampling
- 🔄 Custom VAE (Variational Autoencoder) for encoding and decoding images
- 📝 CLIP (Contrastive Language-Image Pre-Training) for text encoding
- 🚀 Efficient implementation with PyTorch

## Requirements

- Python 3.7+
- PyTorch 1.9+
- transformers
- Pillow
- numpy
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stable-diffusion-pytorch.git
   cd stable-diffusion-pytorch
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the Stable Diffusion v1.5 weights (e.g., `v1-5-pruned-emaonly.ckpt`) and place them in the `models` directory.

4. Download the CLIP tokenizer files (`tokenizer_vocab.json` and `tokenizer_merges.txt`) and place them in the `tokenizer` directory.

## Usage

Here's a basic example of how to use the Stable Diffusion model for image generation:

```python
from stable_diffusion import generate, preload_models_from_standard_weights
from transformers import CLIPTokenizer
from PIL import Image


tokenizer = CLIPTokenizer("tokenizer/tokenizer_vocab.json", "tokenizer/tokenizer_merges.txt")
models = preload_models_from_standard_weights("models/v1-5-pruned-emaonly.ckpt", device="cuda")

# Generate image
output_image = generate(
    prompt="A serene landscape with mountains and a lake at sunset",
    uncond_prompt="",
    input_image=None,  
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models=models,
    seed=42,
    device="cuda",
    idle_device="cpu",
    tokenizer=tokenizer,
)

Image.fromarray(output_image).save("output.png")
```



## Model Architecture

The Stable Diffusion model consists of several key components:

- CLIP Text Encoder
- U-Net Diffusion Model
- Variational Autoencoder (VAE)

For a detailed explanation of the model architecture, please see our [architecture documentation](docs/architecture.md).

## Customization

You can customize various aspects of the model and generation process:

- Adjust image dimensions by modifying `WIDTH` and `HEIGHT` in `config.py`
- Implement new sampling strategies by extending the `DDPMSampler` class
- Experiment with different model architectures in `vae.py` and `unet.py`

Check out our [customization guide](docs/customization.md) for more information.

## Contributing

We welcome contributions to improve this Stable Diffusion implementation! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and commit them (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This implementation is based on the Stable Diffusion model by [Stability AI](https://stability.ai/)
- Thanks to the [PyTorch](https://pytorch.org/) team for their excellent deep learning framework
- The CLIP model implementation is inspired by the work of [OpenAI](https://openai.com/)

---

If you find this implementation helpful, please consider giving it a star ⭐️ and sharing it with others!
