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
print('3) Incremental with sleep')
print('4) Single pattern with sleep')
rule_choice = input('Enter 1, 2, 3, or 4: ').strip()

LEARNING_RATE = 0.1

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

    def train_incremental(self, patterns, n=0.1, max_epochs=100, epsilon=1e-3):
        for epoch in range(max_epochs):
            W_old = self.W.copy()
            for p in patterns:
                p = p.reshape(-1, 1)
                outer = (p @ p.T) / self.size
                self.W = (1 - n) * self.W + n * outer
                np.fill_diagonal(self.W, 0)
            delta = np.max(np.abs(self.W - W_old))
            if delta < epsilon:
                print(f'Incremental converged after {epoch+1} epochs (Δmax={delta:.2e})')
                break

    def train_incremental_with_sleep(self,
                                     patterns,
                                     n=0.1,
                                     n_sleep=0.01,
                                     max_epochs=50,
                                     sleep_iterations=100,
                                     epsilon=1e-3,
                                     k=0.25):
        def heaviside(u):
            return 1.0 if u > 0 else 0.0

        def hopfield_energy(x_vec, W_mat, p_vec, k_val):
            e_hop = -0.5 * x_vec.T.dot(W_mat).dot(x_vec)
            h_factor = heaviside(x_vec.T.dot(p_vec) / self.size - k_val)
            return e_hop * h_factor

        N = self.size
        P = patterns.shape[0]

        for epoch in range(max_epochs):
            W_before = self.W.copy()
            for p_idx in range(P):
                p = patterns[p_idx].reshape(-1, 1)
                outer = (p @ p.T) / N
                self.W = (1 - n) * self.W + n * outer
                np.fill_diagonal(self.W, 0)

                x = np.random.choice([-1, +1], size=(N,), replace=True)

                for it in range(sleep_iterations):
                    i = np.random.randint(0, N)
                    x_new = x.copy()
                    x_new[i] = -x_new[i]

                    E_old = hopfield_energy(x,     self.W, p.flatten(), k)
                    E_new = hopfield_energy(x_new, self.W, p.flatten(), k)
                    delta_E = E_new - E_old

                    Temp = 0.1 * hopfield_energy(p, self.W, p.flatten(), k) + 1e-10

                    if (delta_E < 0) or (np.random.rand() < np.exp(-delta_E / Temp)):
                        x = x_new 
                        outer_x = np.outer(x, x) / N
                        self.W = (1 - n_sleep) * self.W + n_sleep * outer_x
                        np.fill_diagonal(self.W, 0)

            max_change = np.max(np.abs(self.W - W_before))
            if max_change < epsilon:
                print(f'Inc‐with‐sleep converged after {epoch+1} epochs (Δmax={max_change:.2e})')
                break

    def train_single_pattern_with_sleep(self,
                                        pattern,
                                        n=0.1,
                                        n_sleep=0.07,
                                        max_epochs=10,
                                        sleep_iterations=1000,
                                        epsilon=1e-3,
                                        k=0.25,
                                        n_trials=10000,
                                        max_steps=100):
        N = self.size
        p = pattern.reshape(-1, 1)

        self.W = (p @ p.T) / N
        np.fill_diagonal(self.W, 0)

        def iterations_to_converge(W_mat, x_init, max_steps):
            x = x_init.copy()
            for step in range(1, max_steps + 1):
                x_new = np.sign(W_mat @ x)
                x_new[x_new == 0] = 1
                if np.array_equal(x_new, x):
                    return step
                x = x_new
            return max_steps

        pre_iters = []
        for _ in range(n_trials):
            x0 = np.random.choice([-1, 1], size=(N,), replace=True)
            iters = iterations_to_converge(self.W, x0, max_steps)
            pre_iters.append(iters)
        avg_pre = np.mean(pre_iters)
        print(f'Average iterations to converge (pre‐sleep) over {n_trials} trials: {avg_pre:.2f}')

        def heaviside(u):
            return 1.0 if u > 0 else 0.0

        def hopfield_energy(x_vec, W_mat, p_vec, k_val):
            e_hop = -0.5 * x_vec.T.dot(W_mat).dot(x_vec)
            h_factor = heaviside((x_vec.T.dot(p_vec)) / N - k_val)
            return e_hop * h_factor

        p_vec = p.flatten()
        x = np.random.choice([-1, 1], size=(N,), replace=True)

        for it in range(sleep_iterations):
            i = np.random.randint(0, N)
            x_new = x.copy()
            x_new[i] = -x_new[i]

            E_old = hopfield_energy(x,     self.W, p_vec, k)
            E_new = hopfield_energy(x_new, self.W, p_vec, k)
            delta_E = E_new - E_old

            Temp = 0.1 * hopfield_energy(p_vec, self.W, p_vec, k) + 1e-10

            if (delta_E < 0) or (np.random.rand() < np.exp(-delta_E / Temp)):
                x = x_new
                outer_x = np.outer(x, x) / N
                self.W = (1 - n_sleep) * self.W + n_sleep * outer_x
                np.fill_diagonal(self.W, 0)

        post_iters = []
        for _ in range(n_trials):
            x0 = np.random.choice([-1, 1], size=(N,), replace=True)
            iters = iterations_to_converge(self.W, x0, max_steps)
            post_iters.append(iters)
        avg_post = np.mean(post_iters)
        print(f'Average iterations to converge (post‐sleep) over {n_trials} trials: {avg_post:.2f}')

        print(f'\nComparison:')
        print(f'  Pre‐sleep average convergence steps:  {avg_pre:.2f}')
        print(f'  Post‐sleep average convergence steps: {avg_post:.2f}')


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

if rule_choice == '3':
    print('\nUsing incremental with sleep rule.')
    hopfield.train_incremental_with_sleep(np.array(clean_patterns),
                                          n=LEARNING_RATE)

if rule_choice == '4':
    print('\nUsing single pattern with sleep rule.')
    hopfield.train_single_pattern_with_sleep(clean_patterns[0],
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
