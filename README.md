# nn-labs

Small, from-scratch reproductions of neural network papers and mechanisms — no framework magic, just NumPy and the original equations.

## Contents

### [`hopfield-1982/`](./hopfield-1982)

Reproduces the storage-capacity experiment (Fig. 2) from Hopfield's 1982 PNAS paper *"Neural networks and physical systems with emergent collective computational abilities"*.

- Hebbian storage: `T_ij = sum_s mu_i^s * mu_j^s`, `T_ii = 0`
- Asynchronous update dynamics until convergence
- Measures recall error (Hamming distance) vs. number of stored memories `n`, compared against the paper's analytical error probability (Eq. 10)

Run it:

```bash
cd hopfield-1982
uv run capacity_experiment.py
```

Result (N=100, averaged over 100 trials per `n`):

![capacity curve](./hopfield-1982/capacity_curve.png)

Recall degrades sharply around `n ≈ 15` (i.e. `n ≈ 0.15N`), consistent with the paper's reported capacity of `~0.15N` (later refined to the precise `~0.138N` by Amit, Gutfreund & Sompolinsky, 1985, via the replica method).
