import os
import numpy as np

def make_obvious_cpd_data(
    n_samples=200,
    n_atoms=7,
    timesteps=30,
    dims=6,
    cp=5,
    noise=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, n_atoms, timesteps, dims), dtype=np.float32)
    cpds = np.full(n_samples, cp, dtype=np.int64)

    # dummy relations: [samples, timesteps, edges]
    n_edges = n_atoms * (n_atoms - 1)
    relations = np.zeros((n_samples, timesteps, n_edges), dtype=np.int64)

    for i in range(n_samples):
        base = rng.normal(0, noise, size=(n_atoms, timesteps, dims))

        # regime 1: low values
        base[:, :cp, :] += 0.0

        # regime 2: large shift after changepoint
        base[:, cp:, :] += 3.0

        X[i] = base

    X = X.transpose()

    return X, relations, cpds


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_folder = os.path.join(base_dir, 'cp_obvious')
    os.makedirs(out_folder, exist_ok=True)

    X_train, rel_train, cp_train = make_obvious_cpd_data(n_samples=150)
    X_valid, rel_valid, cp_valid = make_obvious_cpd_data(n_samples=25, seed=43)
    X_test, rel_test, cp_test = make_obvious_cpd_data(n_samples=25, seed=44)

    # Save with same filenames as cp_gdelt
    np.save(os.path.join(out_folder, 'trainFeatures.npy'), X_train)
    np.save(os.path.join(out_folder, 'trainTargets.npy'), cp_train)
    np.save(os.path.join(out_folder, 'trainChangePoints.npy'), cp_train)

    np.save(os.path.join(out_folder, 'validFeatures.npy'), X_valid)
    np.save(os.path.join(out_folder, 'validTargets.npy'), cp_valid)
    np.save(os.path.join(out_folder, 'validChangePoints.npy'), cp_valid)

    np.save(os.path.join(out_folder, 'testFeatures.npy'), X_test)
    np.save(os.path.join(out_folder, 'testTargets.npy'), cp_test)
    np.save(os.path.join(out_folder, 'testChangePoints.npy'), cp_test)

    # Print summary
    for fname in sorted(os.listdir(out_folder)):
        p = os.path.join(out_folder, fname)
        a = np.load(p, allow_pickle=True)
        print(fname, a.shape, getattr(a, 'dtype', None))
