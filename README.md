# CS5494 Topics in Generative AI — Assignment 1

Generative models on MNIST: a **Fully Visible Sigmoid Belief Network**, a **neural
autoregressive model**, and a **GAN**.

Group members: **YaoChong CHEN**, **Junyuan DENG**

---

## 1. Environment

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| Hardware | NVIDIA RTX 3060 (6 GB), CUDA 12.4 |
| Other | numpy, matplotlib (figures only), LaTeX (report) |

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# or, from a mirror:
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/
```

MNIST is downloaded automatically on first run into `data/` (the four IDX `.gz`
files, ~12 MB) from `https://ossci-datasets.s3.amazonaws.com/mnist/`.  No other
data preparation is needed.

## 2. Repository layout

```
cs5494_a1/
├── data/                     # MNIST IDX files (auto-downloaded)
├── src/
│   ├── common.py             # seeding, device, CSV/JSON logging, image grids
│   ├── data.py               # MNIST loader + the three binarisation schemes
│   ├── model_fvsbn.py        # Task 1: FVSBN, ordered and unordered variants
│   ├── model_nade.py         # Task 2: MADE/NADE + GRU baseline
│   ├── model_gan.py          # Task 3: DCGAN / WGAN-GP
│   ├── evaluator.py          # MNIST CNN classifier -> mode-collapse metrics
│   ├── baselines.py          # constant / independent-pixel likelihood baselines
│   ├── train_task1.py        # Task 1 training + sampling
│   ├── train_task2.py        # Task 2 training + sampling
│   ├── train_task3.py        # Task 3 training + collapse monitoring
│   ├── test_nade.py          # brute-force equivalence + causality tests (NADE)
│   ├── test_models.py        # causality tests for every autoregressive model
│   ├── eval_classifier.py    # accuracy of the evaluator CNN
│   ├── fvsbn_sampling_study.py     # Gibbs flip-rate diagnostics
│   ├── fvsbn_improve_sampling.py   # annealing / mode-selection experiments
│   ├── run_ablations.py      # ablation driver + table collection
│   ├── rebuild_progress.py   # rebuilds the 5x4 "training progress" figures
│   ├── make_figures.py       # every figure used in the report
│   └── build_report.py       # generates report/numbers.tex and the tables
├── results/                  # logs (.csv), metrics (.json), sample grids (.png)
├── figures/                  # report figures
├── report/                   # LaTeX sources + compiled PDF
└── readme.txt                # run instructions (required by the assignment)
```

## 3. How to run

All commands are run from the `cs5494_a1/` directory.

```bash
# ---- Task 1: Fully Visible Sigmoid Belief Network (binarized MNIST)
# ordered conditionals: exact likelihood + exact ancestral sampling
python src/train_task1.py --model ordered --epochs 40 --out results/task1
# unordered conditionals: pseudo-likelihood + Gibbs sampling (comparison)
python src/train_task1.py --model full --epochs 40 --rank 0 --n-sweeps 200 \
                          --block-size 32 --out results/task1_fullvis

# ---- Task 2: neural autoregressive model (binarized MNIST)
python src/train_task2.py --model nade --fixed-order --epochs 30 \
                          --batch-size 64 --out results/task2_made
python src/train_task2.py --model nade --epochs 50 --batch-size 64 \
                          --out results/task2
python src/train_task2.py --model gruar --epochs 40 --hidden 512 \
                          --out results/task2_gruar

# ---- Task 3: GAN on the original (grayscale) MNIST
python src/train_task3.py --objective dcgan --epochs 45 --train-size 60000 \
                          --out results/task3_dcgan
python src/train_task3.py --objective wgan  --epochs 25 --train-size 60000 \
                          --out results/task3_wgan

# ---- baselines, correctness tests, ablations, figures, report
python src/baselines.py --out results/baselines.json
python src/test_models.py          # causality of every autoregressive model
python src/test_nade.py            # brute-force equivalence of NADE
python src/run_ablations.py --task 1
python src/run_ablations.py --task 2
python src/make_figures.py
python src/build_report.py
cd report && pdflatex main.tex && pdflatex main.tex
```

Every training script accepts `--device auto|cpu|cuda`, `--seed`, `--out` and
`--resume` (per-epoch checkpoint).  On a Colab notebook prefix the commands with
`!python` and make sure the runtime is set to **GPU**.

See `readme.txt` for the complete instructions.

## 4. What each task does

**Task 1 — FVSBN.**  A fully visible model with no hidden units.  We implement
*both* formulations at identical parameter count:

* **unordered** — every pixel conditioned on all the others; trained by
  pseudo-likelihood, sampled with parallel block Gibbs.  Reaches a very low
  pseudo-NLL (0.0846 bits/dim) but its Gibbs chain settles into a noisy mode:
  the fitted conditionals are not mutually consistent, so there is no joint
  distribution to sample from.
* **ordered** — each pixel conditioned on its predecessors, giving a proper
  directed model with an *exact* likelihood (0.1430 bits/dim) and *exact*
  ancestral sampling.  The samples are recognisable digits.

**Task 2 — neural autoregressive model.**  A MADE with two hidden layers.
Trained with the fixed raster order (0.1269 bits/dim) and, alternatively, with
order-agnostic random orderings (NADE, 0.1696 bits/dim).  The critical
implementation detail is that the random ordering is applied to the **spatial
index** of the pixels (per-example connectivity masks) and **not** to the data —
permuting the data silently reduces the model to a constant predictor.

**Task 3 — GAN.**  A DCGAN-style convolutional generator/discriminator on the
*original* grayscale MNIST, plus a WGAN-GP with the same architecture.  Mode
collapse is quantified with an independently trained MNIST classifier
(98.7% test accuracy): the DCGAN covers 9/10 digit classes with a normalised
mode entropy of 0.859 and the digit 1 almost absent, i.e. partial mode collapse
that is invisible in the loss curves.

## 5. Reproducibility

All scripts fix Python/NumPy/PyTorch seeds (`--seed`, default 0).  The random
orderings used by NADE and the latent vectors used by the GAN are drawn from
explicit `torch.Generator` objects.  Likelihood numbers are evaluated on a
*fixed* binarisation of the test set (random threshold, seed 12345) with the
natural pixel order, so they are exactly reproducible; the only residual
non-determinism comes from non-deterministic CUDA kernels.
