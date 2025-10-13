🧠 AI Image Generator

This project demonstrates a text-to-image generation model using the Stable Diffusion 2.1 pipeline from Hugging Face’s Diffusers library. It allows users to generate high-quality, realistic images from textual prompts — showcasing the power of generative AI and deep learning.

📌 Project Overview

The notebook implements an AI Image Generator capable of transforming textual descriptions into detailed and artistic images using Stable Diffusion.
It leverages the latest pre-trained diffusion model (v2.1) for efficient and high-resolution generation with GPU support.

🚀 Features

🧩 Uses Stable Diffusion 2.1 for image synthesis

⚡ GPU-accelerated inference with PyTorch and CUDA

🖼️ Generates high-quality images from text prompts

🧠 Built on top of Hugging Face Diffusers

🔍 Easy customization for prompt, width, and height

🛠️ Tech Stack

| Category      | Tools / Libraries                                                               |
| ------------- | ------------------------------------------------------------------------------- |
| **Language**  | Python                                                                          |
| **Framework** | PyTorch                                                                         |
| **Libraries** | Diffusers, Transformers, Accelerate, BitsAndBytes, SciPy, Safetensors, Xformers |
| **Platform**  | Google Colab / GPU-enabled environment                                          |

📘 How It Works

Install dependencies

!pip install --upgrade diffusers transformers accelerate torch bitsandbytes scipy safetensors xformers


Load the Stable Diffusion pipeline

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)


Enable GPU and configure scheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


Generate an image from a text prompt

prompt = "a house in front of the ocean"
image = pipe(prompt, width=1000, height=1000).images[0]


Display the generated image

import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis("off")
plt.show()

📊 Results

The model successfully generates high-quality AI-generated images from descriptive prompts.
You can modify:

The prompt for different concepts or art styles

The image resolution (width, height)

The model version for varied artistic outcomes

🔮 Future Improvements

Add a user interface using Streamlit or Gradio

Integrate prompt history and image saving options

Experiment with other diffusion models for diverse styles

Optimize performance with quantization and batch generation

🤝 Contributing

Contributions are welcome!
If you’d like to improve the notebook, optimize performance, or add UI features — fork this repo and submit a pull request.

📌 Author

👤 Abrar Jamil

📍 Data Science and Engineering Student, University of Frontier Technology, Bangladesh.

🔗 LinkedIn: https://www.linkedin.com/in/abrarjamil11/

📧 abrarjamil5263@gmail.com
