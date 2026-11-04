# Lightweight deep learning for tuberculosis screening: benchmark code and results

Code, per-run results and analysis scripts for the manuscript

> M. Kumar, "Benchmarking Lightweight Deep Learning Models for Tuberculosis Screening from Chest X-rays in
> Resource-Limited Settings: Accuracy, Efficiency, and Cross-Dataset Generalization."

The study trains six lightweight CNNs/hybrids (MobileNetV3-Large, EfficientNet-B0, EfficientNet-Lite0, MobileViT-S,
ShuffleNetV2, SqueezeNet 1.1; ResNet50 as a reference in the pilot) on TBX11K and tests them internally and on the
Shenzhen and Montgomery datasets, with efficiency measurements, occlusion tests, leave-one-dataset-out (LODO) training,
masked-training mitigation, and local threshold recalibration.

**No images are redistributed.** Obtain the datasets from their original sources and respect their licences:

| Dataset | Used as | Source used in this work |
|---|---|---|
| TBX11K | training / internal test | Kaggle redistribution `vbookshelf/tbx11k-simplified` (original: Liu et al., CVPR 2020) |
| Shenzhen (662 images) and Montgomery (138 images) | external test | Kaggle `kmader/pulmonary-chest-xray-abnormalities` (original: Jaeger et al., QIMS 2014) |

## Contents

```
notebooks/
  phase1_pilot_benchmark_executed.ipynb   3-seed pilot, 7 models, executed output included (Kaggle, T4 GPU)
  phase2_benchmark.ipynb / .py            10-seed benchmark, image-condition tests, LODO, masked training (switches in Section 1)
results/
  phase1_pilot/                           per-run metrics, efficiency, Grad-CAM audit, saved predictions (+ labels)
  phase2_commit1_lodo_mitigation/         LODO (48 runs) and masked-training mitigation (15 runs): metrics + predictions with labels
  phase2_commit2_base_10seeds/            10-seed baseline (60 runs x 5 image conditions): metrics + predictions with labels
  labels_phase1.npz                       labels of the pilot validation / test sets (see "Notes")
analysis/                                 scripts that produce every table and figure; outputs go to analysis_outputs/
analysis_outputs/                         tables (CSV) and figures (PNG) produced by the scripts
```

## Reproducing the experiments (Kaggle)

1. Create a Kaggle notebook, attach the two datasets above, set the accelerator to a GPU and turn Internet on
   (needed to download ImageNet weights).
2. Import `notebooks/phase2_benchmark.ipynb`, edit the three dataset paths in Section 1 if needed.
3. Run once with `QUICK = True` (about 10 minutes) as a smoke test.
4. Set `QUICK = False` and run with *Save and Run All*, one switch combination per commit:
   * LODO + masked training: `RUN_LODO = True`, `RUN_MITIGATION = True`, `RUN_BASE = False` (about 3.5 GPU hours).
   * 10-seed baseline with image-condition tests: `RUN_BASE = True`, the others `False` (about 4.5 GPU hours).
5. Download the `results_phase2/` folder from the notebook output.

Reported runs used an NVIDIA Tesla T4 GPU (pilot; confirm for the Phase 2 commits), Python 3.13, PyTorch 2.11.0 (CUDA 12.8),
timm 1.0.29. Seeds: 0-9 (baseline), 0-4 (masked training), 0-2 (LODO, pilot). Splits are deterministic (stratified 70/15/15,
`random_state=42`). Training is not bit-for-bit reproducible on GPU, so reruns will differ slightly; the seed-to-seed spread
reported in the paper is the relevant measure of that variability.

## Reproducing the tables and figures (no GPU needed)

```
pip install numpy pandas scipy scikit-learn matplotlib
python analysis/phase1_stats.py               # pilot bootstrap comparisons
python analysis/phase1_recalibration.py       # local threshold recalibration (pilot)
python analysis/phase1_figures.py             # pilot figures
python analysis/phase2_baseline_stats.py      # 10-seed AUCs, seed variance, Mann-Whitney + Holm, masking effects
python analysis/phase2_final_tables_figures.py
python analysis/phase2_extra_analysis.py      # LODO table, mitigation, border dependence, flagged fractions
python analysis/paper_fig2_legend.py          # Fig. 2 as printed in the paper
```

Run them in this order; each writes to `analysis_outputs/`.

## Mapping to the manuscript

| Paper item | Source |
|---|---|
| Table I (datasets, splits) | printed in Sections 2-4 of the notebooks |
| Table II (efficiency + AUC, 10 seeds) | `analysis_outputs/table1_main_10seeds.csv` |
| Table III (local recalibration, pilot) | `analysis_outputs/recalibration.csv` (balanced accuracy = `src_bacc`, `loc_bacc`) |
| Table IV (LODO) | `analysis_outputs/lodo_table.csv` |
| Table V (image masking) | `analysis_outputs/border_dependence_internal.csv`, `table2_masking.csv` |
| Table VI (masked-training mitigation) | `analysis_outputs/mitigation_table.csv` |
| Table VII (Grad-CAM border ratio, pilot) | `results/phase1_pilot/gradcam_border_audit.csv` |
| Fig. 1 | `analysis_outputs/fig1_seed_variance_10seeds.png` |
| Fig. 2 | `analysis_outputs/fig2_tradeoff_10seeds_legend.png` |
| Fig. 3 | `analysis_outputs/fig3_masking_effect.png` |
| Fig. 4 | `results/phase1_pilot/figs/gradcam_panel.png` |

## Notes and limitations

* The pilot prediction files did not store labels; `labels_phase1.npz` was reconstructed from the printed split manifest and
  validated by recomputing every recorded AUC (agreement to 1e-16). Phase 2 prediction files contain their labels.
* Efficiency numbers (parameters, FLOPs, size, CPU latency) were measured once, in the pilot run, on a Kaggle Intel Xeon
  2.00 GHz CPU; weights do not affect them.
* TBX11K has no patient identifiers, so patient-level splitting was not possible.
* Model checkpoints are not included (size); the notebooks regenerate them.

## Disclosure

The author used Claude (Anthropic) to assist with drafting the manuscript and writing the code and analysis scripts in this
repository, and reviewed and verified all results.

## Licence

Code: MIT (see `LICENSE`). The datasets are subject to their own licences.
