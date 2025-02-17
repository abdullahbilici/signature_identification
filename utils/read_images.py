import numpy as np
from PIL import Image, ImageFilter
import os

all_alphabets = ".\data\images_background"
dct = dict()

counter = 0
for a in os.listdir(all_alphabets):
    chars_path = os.path.join(all_alphabets,a)
    for char in os.listdir(chars_path):
        temp_char_path = os.path.join(chars_path, char)
        
        dct[counter] = list()
        for image in os.listdir(temp_char_path):
            temp_image_path = os.path.join(temp_char_path, image)
            with Image.open(temp_image_path) as img:
                img = img.convert('L')
                blurred_image = img.filter(ImageFilter.GaussianBlur(radius=2))
                resized_image = blurred_image.resize((64, 64), Image.LANCZOS)   
                dct[counter].append(np.array(resized_image))
        counter += 1


