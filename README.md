<div align="center">

# CHaRM: Segmentation-Free Deep Learning Methodology for Accurate 3D Landmark Detection on Complex Intraoral Scans

[![Journal](https://img.shields.io/badge/Expert%20Systems%20with%20Applications-2026-orange.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![Dataset](https://img.shields.io/badge/Dataset-IOSLandmarks--1k-blue.svg)](https://zenodo.org/records/14757224)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/jrodriguezortega/CHaRM/actions/workflows/ci.yml/badge.svg)](https://github.com/jrodriguezortega/CHaRM/actions/workflows/ci.yml)
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

José Rodríguez-Ortega, Francisco Pérez-Hernández, Siham Tabik

</div>

## Abstract

Accurate 3D landmark detection on point clouds of high-variance biological
anatomical structures, such as Intraoral Scans (IOS), remains a challenging
task in computer vision and pattern recognition. Prior methods struggle with
geometric irregularities and require computationally expensive preprocessing
like tooth segmentation. In this work, we present CHaRM (Conditioned Heatmap
Regression Methodology), a segmentation-free, end-to-end deep learning
approach for 3D tooth landmark detection that explicitly models tooth
presence, enabling robust detection on complex IOS with missing teeth. CHaRM
integrates four components: 1) a point cloud encoder, 2) a decoder with a
heatmap regression head, 3) a teeth-presence classification head, and 4) the
novel CHaR module that leverages teeth-presence information to adapt to
missing teeth, improving detection accuracy in complex dentition. Unlike
two-stage workflows that segment teeth before landmarking, CHaRM operates
directly on IOS point clouds, reducing complexity, avoiding error
propagation, and lowering computational cost.

Given the lack of public datasets for landmark detection in 3D IOS, we
evaluated CHaRM with five point cloud learning backbones on IOSLandmarks-1k,
a new dataset of 1,214 annotated 3D dental models. Both the dataset and code
are publicly released to address the scarcity of open data in orthodontics
and foster reproducible research.

CHaRM with PointMLP, named CHaRNet, achieved the best accuracy and
efficiency. We evaluate under a presence-aware protocol in which a landmark
counts as correct only if its supporting tooth is correctly determined
present and its predicted position lies within the 1 mm clinical tolerance.
CHaRNet attains an F1@1 mm of 88.98 on standard dental models and 68.34
macro-averaged across all dentition types. Compared to state-of-the-art
methods (TSMDL and ALIIOS), CHaRM overcomes the best of the two baselines by
a median of 8.75 F1@1 mm points per scan on incomplete dentitions and
provides statistically similar results on complete dentitions. Besides,
CHaRM delivers up to 14.8× faster inference on GPU. This end-to-end approach
streamlines orthodontic workflows, enhances the precision of 3D IOS
analysis, and enables efficient computer-assisted treatment planning.

## Overview

Each scan is subsampled to 10,000 vertices plus one extra *placeholder*
point that stands for absent teeth. The network predicts one heatmap over
these 10,001 points for each of the 80 landmarks of an arch (16 teeth × 5
landmarks). The repository implements five backbones (PointNet++, DGCNN,
Point Transformer, PointNeXt and PointMLP), each in up to three variants:

| Variant | Description |
| --- | --- |
| `base` | Heatmap regression only. A tooth is predicted present when all five of its heatmaps peak on the mesh; otherwise its five landmarks are placed on the placeholder. |
| `postproc` | Heatmap regression plus a teeth-presence head; the predicted presence filters the landmarks after decoding. |
| `charm` | CHaRM: the presence probabilities condition the heatmaps inside the network through the CHaR module. |

CHaRNet is PointMLP with the `charm` variant.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and targets
Linux with Python 3.12 and PyTorch 2.7.1 (CUDA 12.6 wheels).

```bash
git clone https://github.com/jrodriguezortega/CHaRM.git
cd CHaRM
uv sync
```

This installs the `charm` package, its command line tools and the
development tools (pytest, ruff) into `.venv`. Prefix commands with `uv run`
or activate the environment with `source .venv/bin/activate`.

For a machine without a GPU, install the CPU build of PyTorch instead:

```bash
uv sync --no-default-groups --group cpu
```

### Optional CUDA operators

Every point operator (farthest point sampling, ball query, k-nearest
neighbours, grouping and interpolation) has a pure PyTorch implementation,
so all models run without compiling anything. The `cuda-ops` extra builds
the original CUDA kernels, which speed up DGCNN, PointNeXt and Point
Transformer on the GPU. It needs a CUDA 12.x toolkit (`nvcc`) and is installed after the
first `uv sync`:

```bash
uv sync                                     # installs torch and the build tools
CUDA_HOME=/usr/local/cuda-12.6 uv sync --extra cuda-ops
```

Set `TORCH_CUDA_ARCH_LIST` (for example `"8.6;8.9"`) to compile only for
your GPUs. Run later syncs with `--extra cuda-ops` as well, or `uv` will
remove the compiled packages.

## Dataset

Download [IOSLandmarks-1k](https://zenodo.org/records/14757224) and extract
it to `data/IOSLandmarks-1k`:

```
data/IOSLandmarks-1k/
├── meshes/                 {scan_id}.vtp
├── annotations/json/       {scan_id}.json
├── annotations/fcsv/       {scan_id}.fcsv
├── splits/                 train.txt, val.txt, test.txt
└── labels_schema.json
```

Then build the training samples:

```bash
uv run charm-preprocess
```

This writes one `.npz` file per scan to `data/processed/sigma2_points_10001`
(6.7 GB). Each file holds the subsampled points, their normals, the
landmarks and the Gaussian heatmap target of every landmark (standard
deviation 2 mm). Training scans also get an x-mirrored copy, used as data
augmentation. See `charm-preprocess --help` for the options (dataset and
output folders, number of points, heatmap standard deviation, seed and
number of workers).

### A note on the data splits

We are sorry to report a mistake in the splits used for the experiments of
the paper. IOSLandmarks-1k contains the upper and the lower arch of most
cases, and the two scans of a case should always be in the same split. In
the original splits they were not: for 67 of the 783 cases, the upper and
lower arches were assigned to different splits, so part of the information
about those cases was seen during training and again during validation or
testing. We apologize for this oversight and thank you for your
understanding.

The splits published with the dataset, and used by this repository, fix the
problem. Every case is assigned as a whole to a single split, and the
number of scans of each dentition type in each split is the same as in the
original splits (850 / 182 / 182 scans). Among all assignments meeting
these conditions, we chose the one that moves the fewest scans, which moves
one scan of each of the 67 affected cases.

Because the splits changed, the numbers you obtain with this code may
differ from those reported in the paper. We expect the difference to be
small, and the conclusions of the paper to hold. Only 67 of the 1,214
scans change split, about 5.5%, and the split sizes and the number of
scans of each dentition type stay the same. For each affected case only
one arch moves, so what the model could see ahead of time was the other
arch of the same patient, a different scan, and never the evaluated scan
itself. The comparisons in the paper are between methods trained and
tested on one common split, so a small change in which scans sit where
moves every method together.

## Training

Training is configured with [Hydra](https://hydra.cc/); the configuration
files are in [`configs/`](configs). Every experiment of the paper has a
configuration in `configs/experiment/`:

| Backbone | Experiments |
| --- | --- |
| PointNet++ | `pointnet2_base`, `pointnet2_charm` |
| DGCNN | `dgcnn_base`, `dgcnn_charm` |
| Point Transformer | `pointtransformer_base`, `pointtransformer_charm` |
| PointNeXt | `pointnext_base`, `pointnext_charm` |
| PointMLP | `pointmlp_base`, `pointmlp_postproc`, `pointmlp_charm` (alias `charnet`) |

```bash
uv run charm-train experiment=charnet
```

Each run writes its configuration, log and checkpoints to
`outputs/<experiment>/<date>_<time>/`. The checkpoint with the best
validation `f1_1mm_macro` is kept along with the last one, and is evaluated
on the test split at the end of training (disable with `test=false`).

Any configuration value can be overridden from the command line:

```bash
# Heatmaps of standard deviation 1 mm, then a different heatmap loss weight.
uv run charm-preprocess --sigma 1
uv run charm-train experiment=charnet paths.processed_dir=data/processed/sigma1_points_10001 loss.weight_seg=0.01

# Train on the CPU, or resume an interrupted run.
uv run charm-train experiment=pointnet2_charm trainer=cpu
uv run charm-train experiment=charnet ckpt_path=outputs/charnet/<run>/checkpoints/last.ckpt

# Sweep several experiments (Hydra multirun).
uv run charm-train -m experiment=pointmlp_base,pointmlp_postproc,pointmlp_charm
```

Training batches contain a fixed number of scans of each dentition type,
set by `data.samples_per_dentition_type` (one value per type, in the order
`00 01 02 03 04 10 11 12 13 14`), so rare dentition types are seen as often
as common ones.

To change the number of points, preprocess the dataset again and point the
training to the new folder (PointMLP also needs the number of points):

```bash
uv run charm-preprocess --num-points 16385
uv run charm-train experiment=charnet paths.processed_dir=data/processed/sigma2_points_16385 model.points=16385
```

### Experiment tracking

[MLflow](https://mlflow.org/) is the default logger. Runs are stored in
`./mlruns`, in the experiment `charm`, under the name of the experiment
configuration. To browse them, start the MLflow UI from the repository root
and open <http://127.0.0.1:5000>:

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

To log to an MLflow tracking server instead, set its address:

```bash
export MLFLOW_TRACKING_URI=http://<host>:5000
uv run charm-train experiment=charnet
```

CSV and TensorBoard loggers are also available, alone or combined:

```bash
uv run charm-train experiment=charnet logger=csv
uv run charm-train experiment=charnet logger=tensorboard
uv run charm-train experiment=charnet "logger=[mlflow,tensorboard]"
uv run tensorboard --logdir outputs
```

## Evaluation

```bash
uv run charm-eval ckpt_path=outputs/charnet/<run>/checkpoints/<best>.ckpt
```

The checkpoint is scored on the test split (`split=val` for validation) with
the presence-aware protocol of the paper. A predicted landmark is a true
positive when it is predicted on the scan, its tooth is present in the
ground truth, and it lies within 1 mm of the annotation. Precision divides
the true positives by the landmarks predicted on the scan, recall by the
annotated landmarks, and F1@1 mm is their harmonic mean. Results are
reported per dentition type and averaged over types (macro: mean precision
and recall, then their harmonic mean) or over all landmarks (micro). The
mean Euclidean distance error (MEDE, mm) is computed over landmarks that are
both predicted and annotated, and the teeth-presence F1 over teeth.

Results are written to `outputs/eval/<date>_<time>/`:

| File | Content |
| --- | --- |
| `per_type_<split>.csv` | Precision, recall, F1@1 mm, MEDE and presence F1 per dentition type, plus macro and micro averages |
| `metrics_<split>.json` | The same scores and the evaluation settings |
| `per_scan_<split>.csv` | Scores of every scan |
| `predictions/` | Predicted landmarks of every scan, as JSON and FCSV files |

Use `threshold_mm=2` to change the distance threshold, `device=cpu` to run
on the CPU, and `presence_gate=false` to decode every landmark from its own
heatmap instead of from the predicted teeth presence.

## Inference on new scans

```bash
uv run charm-predict <checkpoint.ckpt> scan_upper.stl scans/ -o predictions
```

`charm-predict` reads `.vtp`, `.stl`, `.ply` and `.obj` meshes, given as
files or folders, with coordinates in millimetres. The arch is read from
the `_upper` or `_lower` suffix of the file name; pass `--arch upper` or
`--arch lower` otherwise. For each scan it writes:

- `<name>.json`, in the annotation format of IOSLandmarks-1k
  (`labels_schema.json`);
- `<name>.fcsv`, a 3D Slicer markups file that can be loaded on top of the
  mesh.

Only landmarks of teeth predicted present are written. Inference runs on the
first GPU when available (`--device cpu` to force the CPU). To measure the
preprocessing and inference time, add `--benchmark`, which runs warm-up
iterations followed by timed runs (`--warmup`, `--runs`) and reports their
mean and standard deviation.

## Repository structure

```
configs/                 Hydra configurations (train.yaml, eval.yaml and groups)
packages/                Optional CUDA operators (cuda-ops extra)
src/charm/
├── cli/                 charm-preprocess, charm-train, charm-eval, charm-predict
├── data/                Preprocessing, dataset and dentition-balanced datamodule
├── models/              Backbones, CHaR module and point operators
├── lit_module.py        Lightning module
├── losses.py            Heatmap and teeth-presence losses
├── metrics.py           Presence-aware metrics
├── postprocess.py       Heatmap decoding
└── io.py                JSON and FCSV writers
tests/                   CPU tests (uv run pytest)
```

## Citation

If you use this code or IOSLandmarks-1k, please cite:

```bibtex
@article{rodriguez2026charm,
  title={CHaRM: Segmentation-Free Deep Learning Methodology for Accurate 3D Landmark Detection on Complex Intraoral Scans},
  author={Rodr{\'\i}guez-Ortega, Jos{\'e} and P{\'e}rez-Hern{\'a}ndez, Francisco and Tabik, Siham},
  journal={Expert Systems with Applications},
  pages={134592},
  year={2026},
  publisher={Elsevier}
}
```

## License

The code is released under the [MIT license](LICENSE). Code adapted from
other projects keeps its original license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The IOSLandmarks-1k
dataset is distributed separately under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
