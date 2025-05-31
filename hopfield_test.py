import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 28
CLEAN_DIR = 'clean'
NOISY_DIR = 'noisy'

print('Choose learning rule:')
print('1) One-shot (Hebbian, all patterns at once)')
print('2) Incremental (cycle through all patterns each epoch)')
print('3) Deep narrow well (randomized, selective update)')
print('4) Sequential incremental (train on one pattern until convergence, then next)')
rule_choice = input('Enter 1, 2, 3, or 4: ').strip()

LEARNING_RATE = 0.001

if rule_choice == '3':
    kappa = float(input('Enter kappa (e.g., 10): '))
    n_samples = int(input('Enter number of random samples per pattern (e.g., 100): '))

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def train_oneshot(self, patterns):
        for p in patterns:
            p = p.reshape(-1, 1)
            self.W += p @ p.T
        np.fill_diagonal(self.W, 0)
        self.W /= len(patterns)

    def train_incremental(self, patterns, n=LEARNING_RATE, max_epochs=100, epsilon=1e-3):
        for epoch in range(max_epochs):
            W_old = self.W.copy()
            for p in patterns:
                p = p.reshape(-1, 1)
                self.W = (1 - n) * self.W + n * (p @ p.T)
                np.fill_diagonal(self.W, 0)
            if np.max(np.abs(self.W - W_old)) < epsilon:
                print(f'Incremental learning converged after {epoch+1} epochs')
                break

    def train_deep_narrow(self, patterns, kappa, n_samples, max_epochs=100, epsilon=1e-3, n=LEARNING_RATE):
        for i, p in enumerate(patterns):
            for epoch in range(max_epochs):
                W_old = self.W.copy()
                p = p.reshape(-1, 1)
                p_flat = p.flatten()
                update_count = 0
                for _ in range(n_samples):
                    x = np.random.choice([-1, 1], size=p_flat.shape)
                    if abs(np.dot(x, p_flat)) - kappa > 0:
                        self.W = (1 - n) * self.W + n * (p @ p.T)
                        np.fill_diagonal(self.W, 0)
                        update_count += 1
                if np.max(np.abs(self.W - W_old)) < epsilon:
                    print(f'Deep narrow well learning for pattern {i} converged after {epoch+1} epochs, updates: {update_count}')
                    break

    def recall(self, pattern, steps=5):
        s = pattern.copy()
        for _ in range(steps):
            s = np.sign(self.W @ s)
            s[s == 0] = 1
        return s

    def train_sequential_incremental(self, patterns, n=LEARNING_RATE, max_epochs=100, epsilon=1e-3):
        for i, p in enumerate(patterns):
            for epoch in range(max_epochs):
                W_old = self.W.copy()
                p = p.reshape(-1, 1)
                self.W = (1 - n) * self.W + n * (p @ p.T)
                np.fill_diagonal(self.W, 0)
                if np.max(np.abs(self.W - W_old)) < epsilon:
                    print(f'Sequential incremental learning for pattern {i} converged after {epoch+1} epochs')
                    break


def load_image_as_pattern(path):
    img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)
    arr = np.where(arr == 255, 1, -1).astype(np.int8)
    return arr.flatten()


def show_two_images(noisy, recalled, title1='Noisy Input', title2='Recalled Output'):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(noisy.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axs[0].set_title(title1)
    axs[0].axis('off')
    axs[1].imshow(recalled.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axs[1].set_title(title2)
    axs[1].axis('off')
    plt.show()


TRAIN_DIGITS = [1, 8]

clean_files = sorted([
    os.path.join(CLEAN_DIR, f)
    for f in os.listdir(CLEAN_DIR)
    if f.endswith('.png')
])
clean_files = [
    f for f in clean_files
    if int(os.path.splitext(os.path.basename(f))[0]) in TRAIN_DIGITS
]
noisy_files = sorted([
    os.path.join(NOISY_DIR, f)
    for f in os.listdir(NOISY_DIR)
    if f.endswith('.png')
])

clean_patterns = [load_image_as_pattern(f) for f in clean_files]
clean_labels = [int(os.path.splitext(os.path.basename(f))[0]) for f in clean_files]

hopfield = HopfieldNetwork(IMG_SIZE * IMG_SIZE)
if rule_choice == '4':
    print('Using sequential incremental learning rule.')
    hopfield.train_sequential_incremental(np.array(clean_patterns), n=LEARNING_RATE)
elif rule_choice == '3':
    print('Using deep narrow well learning rule.')
    hopfield.train_deep_narrow(np.array(clean_patterns), kappa=kappa, n_samples=n_samples)
elif rule_choice == '2':
    print('Using incremental learning rule.')
    hopfield.train_incremental(np.array(clean_patterns), n=LEARNING_RATE)
else:
    print('Using one-shot Hebbian learning rule.')
    hopfield.train_oneshot(np.array(clean_patterns))

print('Available noisy images (only for trained digits):')
filtered_noisy_files = [
    f for f in noisy_files
    if int(os.path.basename(f).split('_')[0]) in TRAIN_DIGITS
]
for idx, f in enumerate(filtered_noisy_files):
    print(f'[{idx}] {os.path.basename(f)}')

while True:
    choice = input("Select a noisy image by index (or 'q' to quit): ").strip()
    if choice.lower() == 'q':
        break
    try:
        idx = int(choice)
        if idx < 0 or idx >= len(filtered_noisy_files):
            print('Invalid index. Try again.')
            continue
    except ValueError:
        print('Invalid input. Enter a number or q to quit.')
        continue
    noisy_pattern = load_image_as_pattern(filtered_noisy_files[idx])
    recalled_pattern = hopfield.recall(noisy_pattern)
    show_two_images(noisy_pattern, recalled_pattern) 