import os
import numpy as np

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cp_obvious')
files = ['trainFeatures.npy', 'validFeatures.npy', 'testFeatures.npy']

for f in files:
    p = os.path.join(folder, f)
    a = np.load(p, allow_pickle=True)
    print('Before', f, a.shape, a.dtype)
    # expected current shape: (n_samples, n_atoms, timesteps, dims)
    if a.ndim == 4:
        a2 = a.transpose(0, 2, 3, 1)  # -> (n_samples, timesteps, dims, n_atoms)
    else:
        raise RuntimeError(f'Unexpected ndim for {f}: {a.ndim}')
    a2 = a2.astype(np.float64)
    np.save(p, a2)
    b = np.load(p, allow_pickle=True)
    print('After', f, b.shape, b.dtype)

# show changepoints/targets shapes
for f in sorted(os.listdir(folder)):
    if f.endswith('ChangePoints.npy') or f.endswith('Targets.npy'):
        a = np.load(os.path.join(folder, f), allow_pickle=True)
        print(f, a.shape, a.dtype)
