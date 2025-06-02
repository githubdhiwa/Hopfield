import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 28
N = IMG_SIZE * IMG_SIZE
CLEAN_DIR = 'clean'
NOISY_DIR = 'noisy'

print('Choose learning rule:')
print('1) One-shot (Hebbian, all patterns at once)')
print('2) Incremental (cycle through all patterns each epoch)')
print('3) Deep narrow well (randomized, selective update)')
print('4) Sequential incremental (train on one pattern until convergence, then next)')
rule_choice = input('Enter 1, 2, 3, or 4: ').strip()

LEARNING_RATE = 0.3

if rule_choice == '3':
    kappa = float(input('Enter kappa for deep-narrow (e.g., 56): '))
    n_samples = int(input('Enter # random samples per pattern (e.g., 300): '))

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def train_oneshot(self, patterns):
        self.W.fill(0.0)
        for p in patterns:
            p = p.reshape(-1, 1)
            outer = (p @ p.T) / self.size
            self.W += outer
        np.fill_diagonal(self.W, 0)

    def train_incremental(self, patterns, n=LEARNING_RATE, max_epochs=100, epsilon=1e-3, n_samples=1000, kappa=100):
        for epoch in range(max_epochs):
            W_old = self.W.copy()
            for p in patterns:
                p = p.reshape(-1, 1)
                outer = (p @ p.T) / self.size
                self.W = (1 - n) * self.W + n * outer
                np.fill_diagonal(self.W, 0)

                p_flat = p.flatten()
                x = np.random.choice([-1, 1], size=(self.size,))
                # dot_prod = np.dot(x, p_flat)
                for _ in range(n_samples):
                    
                    delta_E = E_new - E_old

            delta = np.max(np.abs(self.W - W_old))

            # if delta < epsilon:
            #     print(f'Incremental converged after {epoch+1} epochs (Δmax={delta:.2e})')
            #     break

    def train_deep_narrow(self, patterns, kappa, n_samples, max_epochs=100, epsilon=1e-3, n=LEARNING_RATE):
        for i, p in enumerate(patterns):
            p = p.reshape(-1, 1)
            p_flat = p.flatten()
            print(f'\n--- Deep-Narrow: training pattern {i} ---')
            for epoch in range(max_epochs):
                W_old = self.W.copy()
                sum_updates = np.zeros_like(self.W)
                hits = 0
                for _ in range(n_samples):
                    x = np.random.choice([-1, 1], size=(self.size,))
                    dot_prod = np.dot(x, p_flat)
                    if dot_prod > kappa:
                        outer = (p @ p.T) / self.size
                        sum_updates += outer
                        hits += 1
                if hits > 0:
                    avg_update = sum_updates / hits
                    self.W = (1 - n) * self.W + n * avg_update
                    np.fill_diagonal(self.W, 0)
                delta = np.max(np.abs(self.W - W_old))
                print(f'[Pattern {i}] Epoch {epoch+1:3d} | hits = {hits:4d} | ΔW_max = {delta:.2e}')
                if delta < epsilon:
                    print(f'→ Converged on pattern {i} after {epoch+1} epochs (hits={hits})')
                    break
            else:
                print(f'→ Reached max_epochs ({max_epochs}) on pattern {i} without full convergence')

    def train_sequential_incremental(self, patterns, n=LEARNING_RATE, max_epochs=100, epsilon=1e-3):
        for i, p in enumerate(patterns):
            p = p.reshape(-1, 1)
            print(f'\n--- Sequential Incremental: training pattern {i} ---')
            for epoch in range(max_epochs):
                W_old = self.W.copy()
                outer = (p @ p.T) / self.size
                self.W = (1 - n) * self.W + n * outer
                np.fill_diagonal(self.W, 0)
                delta = np.max(np.abs(self.W - W_old))
                if delta < epsilon:
                    print(f'[Pattern {i}] Converged after {epoch+1} epochs (Δmax={delta:.2e})')
                    break
            else:
                print(f'[Pattern {i}] Did NOT converge within {max_epochs} epochs (Δmax={delta:.2e})')        
    
    def train_With_sleep(self, patterns, kappa, n_samples, max_epochs=100, epsilon=1e-3, n_wake=LEARNING_RATE, n_sleep=LEARNING_RATE):
        for i, p in enumerate(patterns):
            # Wake
            p = p.reshape(-1, 1)
            for epoch in range(max_epochs):
                W_old = self.W.copy()
                outer = (p @ p.T) / self.size
                self.W = (1 - n_wake) * self.W + n_wake * outer
                np.diagonal(self.W, 0)
                delta = np.max(np.abs(self.W - W_old))
                if delta < epsilon:
                    break 
            # Sleep
            p_flat = p.flatten()
            for epoch in range(max_epochs):
                W_old = self.W.copy()
                sum_updates = np.zeros_like(self.W)
                hits = 0
                for _ in range(n_samples):
                    x = np.random.choice([-1, 1], size=(self.size,))
                    dot_prod = np.dot(x, p_flat)
                    if dot_prod > kappa:
                        outer = (p @ p.T) / self.size 
                        sum_updates += outer
                        hits += 1
                self.W = (1 - n_sleep) * self.W + n_sleep * sum_updates
                np.fill_diagonal(self.W, 0)
                delta = np.max(np.abs(self.W - W_old))
                if delta < epsilon:
                    break 

    def recall(self, pattern, steps=5):
        s = pattern.copy()
        for _ in range(steps):
            s = np.sign(self.W @ s)
            s[s == 0] = 1
        return s

def load_image_as_pattern(path):
    img = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)
    arr = np.where(arr == 255, 1, -1).astype(np.int8)
    return arr.flatten()

def show_two_images(noisy, recalled,
                    title1='Noisy Input', title2='Recalled Output'):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(noisy.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axs[0].set_title(title1)
    axs[0].axis('off')
    axs[1].imshow(recalled.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axs[1].set_title(title2)
    axs[1].axis('off')
    plt.tight_layout()
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
clean_patterns = [load_image_as_pattern(f) for f in clean_files]
clean_labels   = [int(os.path.splitext(os.path.basename(f))[0]) for f in clean_files]

hopfield = HopfieldNetwork(N)

if rule_choice == '1':
    print('\nUsing one-shot Hebbian rule.')
    hopfield.train_oneshot(np.array(clean_patterns))

elif rule_choice == '2':
    print('\nUsing incremental Hebbian rule.')
    hopfield.train_incremental(np.array(clean_patterns), n=LEARNING_RATE)

elif rule_choice == '3':
    print('\nUsing deep-narrow well rule.')
    hopfield.train_deep_narrow(np.array(clean_patterns),
                               kappa=kappa,
                               n_samples=n_samples,
                               max_epochs=100,
                               epsilon=1e-3,
                               n=LEARNING_RATE)

else:
    print('\nUsing sequential incremental rule.')
    hopfield.train_sequential_incremental(np.array(clean_patterns),
                                          n=LEARNING_RATE)

print('\nAvailable noisy images (only for trained digits):')
noisy_files = sorted([
    os.path.join(NOISY_DIR, f)
    for f in os.listdir(NOISY_DIR)
    if f.endswith('.png')
])
filtered_noisy_files = [
    f for f in noisy_files
    if int(os.path.basename(f).split('_')[0]) in TRAIN_DIGITS
]
for idx, f in enumerate(filtered_noisy_files):
    print(f'[{idx}] {os.path.basename(f)}')

while True:
    choice = input("\nSelect a noisy image by index (or 'q' to quit): ").strip()
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
    recalled_pattern = hopfield.recall(noisy_pattern, steps=10)
    show_two_images(noisy_pattern, recalled_pattern)
