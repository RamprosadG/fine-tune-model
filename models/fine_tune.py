import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from torchvision import transforms

class ImagePairDataset(Dataset):
    def __init__(self, input_images, output_images, transform=None):
        self.input_images = input_images
        self.output_images = output_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = self.input_images[idx]
        output_img = self.output_images[idx]
        
        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)
        
        return input_img, output_img

def load_data(input_image_dir, output_image_dir, transform=None):
    input_images = [os.path.join(input_image_dir, f) for f in os.listdir(input_image_dir)]
    output_images = [os.path.join(output_image_dir, f) for f in os.listdir(output_image_dir)]
    
    input_image_tensors = [transform(Image.open(img)) for img in input_images]
    output_image_tensors = [transform(Image.open(img)) for img in output_images]
    
    return input_image_tensors, output_image_tensors

def fine_tune():
    # Load the pre-trained model
    model_name = "CompVis/stable-diffusion-v1-4-original"
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")

    # Hyperparameters
    learning_rate = 1e-5
    batch_size = 4
    epochs = 3
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    input_image_dir = "data/raw/input_images/"
    output_image_dir = "data/raw/output_images/"
    
    input_image_tensors, output_image_tensors = load_data(input_image_dir, output_image_dir, transform)
    
    dataset = ImagePairDataset(input_image_tensors, output_image_tensors, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = AdamW(pipe.unet.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        pipe.train()
        for i, (input_img, target_img) in enumerate(dataloader):
            input_img = input_img.to("cuda")
            target_img = target_img.to("cuda")
            
            # Forward pass: Generate output from input
            output = pipe(input_img)["sample"]

            # Compute loss (L2 loss between output and target)
            loss = torch.nn.functional.mse_loss(output, target_img)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item()}")

    # Save the fine-tuned model
    pipe.save_pretrained("models/stable_diffusion/fine_tuned_model")
    print("Model fine-tuned and saved!")

if __name__ == "__main__":
    fine_tune()
