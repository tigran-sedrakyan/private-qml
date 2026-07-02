# Private training in quantum machine learning — code

Companion code for **"Private training in quantum machine learning"**. The paper studies whether hybrid variational quantum classifiers admit a privacy–utility advantage over matched classical baselines when both are trained with the same classical DP-SGD mechanism — per-example clipping at threshold `C`, calibrated Gaussian noise with multiplier `σ`, accounted via subsampled Rényi DP through Opacus.

Two experimental studies are performed:

- `tabular/` — synthetic 4-class classification.
- `image/` — Honda Scenes weather/time-of-day classification via the BAE quantum image-loading pipeline.

Both use PyTorch + [Opacus](https://opacus.ai/) for DP-SGD accounting and per-example gradients, [PennyLane](https://pennylane.ai/) for the parameterized quantum circuits, and [deel-torchlip](https://github.com/deel-ai/deel-torchlip) for the gradient-norm-preserving Lipschitz baseline.

## Layout

```
.
├── tabular/
│   ├── privacy_torch_heatmap.ipynb    # clip × B and clip × ε heatmap sweeps
│   ├── privacy_torch_clipped.ipynb    # per-epoch clipping diagnostics (σ = 0)
│   ├── privacy_torch_private.ipynb    # per-epoch MSE under DP-SGD (ε = 10)
│   └── experiment_summaries/          # pickled run summaries consumed by the plot cells
└── image/
    ├── classical_model.ipynb          # SimpleCNN baseline on 64×96 gray images
    ├── lipschitz_model.ipynb          # Lipschitz baseline
    ├── quantum_model.ipynb            # BAE loader + variational classifier (hybrid, DP-SGD)
    ├── common.py                      # shared datasets, criterion, train/test loops
    ├── loaders/
    │   ├── 4class_2000/               # BAE qasm circuits
    │   └── original_data_4class_2000/ # source images grouped by class
    └── results/                       # aggregated image-classifier runs
        ├── 4class_best.json           # {quantum,classical}×{private,nonprivate} best-run traces (10 seeds × 50 epochs)
        └── visualize.ipynb            # produces the accuracy / delta-L plots
```

## `tabular/` — synthetic clipping and DP-SGD study

The task is `sklearn.datasets.make_classification` with 16 features, 4 classes, 2048 samples and `flip_y = 0.1` (non-separable enough to make the metric differences readable). Three matched model families are trained through the same DP-SGD pipeline:

- **ClassicalNet** — 4-hidden-unit MLP with `tanh` on every layer; `tanh` activations are used because tempered-sigmoid activations empirically outperform ReLU under DP-SGD.
- **QuantumNet** — amplitude encoding on `⌈log₂ 16⌉ = 4` qubits, `L` layers of `RZ-RY-RZ` per qubit followed by a CNOT ring, Pauli-`Z` expectations as logits. `L` is auto-tuned so the parameter count matches the classical net.
- **LipschitzNet** — same MLP topology built from `deel.torchlip.SpectralLinear` (k = 1) + `GroupSort2`, with the output layer absorbing a Lipschitz constant `K = C` matched to the DP-SGD clipping threshold.

### `privacy_torch_heatmap.ipynb`
Builds the two heatmaps:

- **Heatmap 1** (clip threshold `C ∈ {0.5, 1.5, 4.5, 10.0}` × batch size `B ∈ {32, 64, 128, 512}`, `σ = 0`) — the clipping-only ablation. Isolates the effect of clipping alone by disabling the DP Gaussian noise. Reproduces the observation that `B` barely matters once the LR is tuned, but classical models collapse for `C ≤ 1.5` while quantum and Lipschitz models do not.
- **Heatmap 2** (privacy budget `ε ∈ {0.5, 1.0, 5.0, 10.0}` × clip threshold, `B = 64`, `σ` derived from `ε` via the Opacus RDP accountant) — the private heatmap. Shows that the sweet spot for the quantum model is at small `C` (which keeps `σC` low without sacrificing signal), yielding ~5% accuracy over the best classical configuration at `ε = 10`.

Each cell is 10 seeds × 100 epochs, reporting mean and std of test accuracy.

### `privacy_torch_clipped.ipynb`
Per-epoch clipping diagnostics at the `C = 1.5, B = 64` cell where classical and quantum models diverge in Heatmap 1. Same models, no DP noise. At the end of every epoch it records the pre-clipping per-example gradient tensor at the Polyak-averaged parameters (EMA of θ over the epoch), which the downstream cells convert into informative metrics:

- Population and per-example gradient norms.
- Clipping probability and clipping bias norm.
- Distance from the unclipped loss and directional alignment.

This is where the paper's structural claim is verified empirically: at `C = 1.5` the quantum gradients stay inside the clipping ball on average and in the worst case, so the clipping probability and bias stay small, while the classical model clips heavily and its directional alignment degrades.

### `privacy_torch_private.ipynb`
Same per-epoch diagnostic pipeline as `_clipped`, but with the actual DP-SGD noise on (`ε = 10`, `δ = 10⁻⁵`, `C = 1.5`, `B = 64`). Reports the composite MSE evolution alongside test accuracy — this aggregates the clipping bias, sampling variance, and DP noise contributions per epoch. The classical model's higher MSE is what drives its lower accuracy in this regime, giving the quantum classifier the ~5% edge visible in Heatmap 2.

## `image/` — Honda Scenes / BAE image-classification pipeline

Private training on a real-image task instead of synthetic tabular data. 2000 images sampled from the Honda Scenes dataset, four classes combining weather and time of day (`clear-day`, `snow-day`, `clear-evening`, `snow-evening`), downscaled to `64 × 96` grayscale.

- **`classical_model.ipynb`** — `SimpleCNN`, the classical baseline referenced in the paper: 7 convolutional layers with `GroupNorm(1, C_out)` and `MaxPool2d(2)`, then a `24 → 3 → num_classes` head. Optimized with SGD; when `private = True` it is wrapped by Opacus's `hooks`-mode DP-SGD.
- **`lipschitz_model.ipynb`** — matched topology using `deel.torchlip.SpectralConv2d` (k = 1), `GroupSort2` activations, and `ScaledAvgPool2d`, giving a globally Lipschitz feature extractor.
- **`quantum_model.ipynb`** — the hybrid `BAENet` model based on a loading + classification pipeline. Each image is split into a `2 × 3` grid of `32 × 32` patches; a pre-trained block-amplitude-encoding (BAE) loader (10-qubit qasm circuit) prepares the state for each patch; `L = 3` variational layers of `RZ-RY-RZ` + CNOT ring act on top; the Pauli-`Z` expectation of the first qubit of each patch is measured.

At `ε = 10`, `δ = 10⁻⁵`, `C = 1.5` (the same private configuration as the synthetic study), the quantum model reproduces the paper's headline: higher test accuracy than the CNN and a smaller distance to the non-private loss.