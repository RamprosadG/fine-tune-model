import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms

# Load the fine-tuned model
model_path = "models/stable_diffusion/fine_tuned_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# Define the preprocessing transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def generate_image(input_image_path):
    # Load and preprocess the input image
    input_image = Image.open(input_image_path)
    input_tensor = transform(input_image).unsqueeze(0).to("cuda")
    
    # Generate the output image
    with torch.no_grad():
        output = pipe(input_tensor)["sample"]

    # Convert output tensor back to image and display
    output_image = output.squeeze().cpu()
    output_image_pil = transforms.ToPILImage()(output_image)
    output_image_pil.show()

if __name__ == "__main__":
    # Example: Generate image from an input image
    input_image_path = "data/raw/input_images/test_image.jpg"
    generate_image(input_image_path)
