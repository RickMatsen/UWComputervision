import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def pad_to_square(img_tensor):
    _, h, w = img_tensor.size()
    diff = abs(h - w)
    padding = diff // 2
    if h > w:
        pads = (padding, diff - padding, 0, 0)
    else:
        pads = (0, 0, padding, diff - padding)
    return torch.nn.functional.pad(img_tensor, pads, "constant", 0)

if __name__ == '__main__':
    # Directories
    input_dir = './data/all_images'
    output_dir = './data_unann_padded'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transformation pipeline to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor() # Converts image to tensor
    ])



    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')): # Check for image files
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB') # Open and convert to RGB for consistency
            
            img_tensor = transform(img) # Convert image to tensor
            padded_img_tensor = pad_to_square(img_tensor) # Pad image to make it square

            # Convert tensor back to image
            padded_img = transforms.ToPILImage()(padded_img_tensor)
            
            # Save the padded image
            save_path = os.path.join(output_dir, filename)
            padded_img.save(save_path)

    print("Padding complete. Padded images are saved in", output_dir)
