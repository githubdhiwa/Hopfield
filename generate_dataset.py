import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE = 28
NUM_DIGITS = 10
NOISY_VARIANTS = 5
NOISE_PERCENT = 0.05  # 5% of pixels will be flipped
CLEAN_DIR = 'clean'
NOISY_DIR = 'noisy'

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

# Use default PIL font
try:
    font = ImageFont.truetype("arial.ttf", 22)
except IOError:
    font = ImageFont.load_default()

def create_digit_image(digit):
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(str(digit), font=font)
    draw.text(((IMG_SIZE-w)//2, (IMG_SIZE-h)//2), str(digit), fill=255, font=font)
    # Threshold to binary
    arr = np.array(img)
    arr = np.where(arr > 128, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode='L')

def add_noise(img, noise_percent=NOISE_PERCENT):
    arr = np.array(img)
    flat = arr.flatten()
    n_pixels = flat.size
    n_flip = int(noise_percent * n_pixels)
    flip_indices = np.random.choice(n_pixels, n_flip, replace=False)
    # Flip 0 <-> 255
    flat[flip_indices] = 255 - flat[flip_indices]
    noisy_arr = flat.reshape(arr.shape)
    return Image.fromarray(noisy_arr, mode='L')

for digit in range(NUM_DIGITS):
    clean_img = create_digit_image(digit)
    clean_img.save(os.path.join(CLEAN_DIR, f'{digit}.png'))
    for i in range(NOISY_VARIANTS):
        noisy_img = add_noise(clean_img)
        noisy_img.save(os.path.join(NOISY_DIR, f'{digit}_noisy_{i}.png'))

print('Binary dataset generated: clean/ and noisy/ directories created.') 