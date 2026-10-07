# Indian Food Vision-Language Foundation Model (VFM) Benchmark

A comprehensive benchmark for evaluating Vision-Language Foundation Models (VFMs) on fine-grained Indian Food classification, zero-shot generalization, linear probing, robustness, and deduplication audit.

---

## 📁 Repository Structure

```
indian_food_vfm_benchmark/
├── README.md                              # Main documentation                   
├── notebooks/                             # Primary evaluation & benchmark pipeline
│   ├── food_research_01.ipynb             # Dataset loading, class mapping, and stratified splits
│   ├── food_research_02.ipynb             # Zero-shot performance & linear probing evaluation
│   ├── food_research_04.ipynb             # Synthetic corruption evaluation & Grad-CAM interpretability
│   ├── food_research_06.ipynb             # Perceptual hashing (pHash) & embedding deduplication audit
│   ├── food_research_07.ipynb             # Multi-seed variance assessment & stability evaluation
│   ├── food_research_08.ipynb             # Statistical significance testing & metric aggregation
│   └── food_research_09.ipynb             # Revision tables generation & repro package builder
├── legacy_original_submission/            # Historical initial submission notebooks
│   ├── README.md                          # Documentation for legacy notebooks
│   ├── food_research_03_1.ipynb           # Early exploratory analysis (Part 1)
│   ├── food_research_03_2.ipynb           # Early exploratory analysis (Part 2)
│   └── food_research_05.ipynb             # Preliminary error analysis scripts
└── repro_package/                         # Reproducibility package with pre-computed evaluation data
    ├── class_list.csv                     # Dataset class vocabulary
    ├── splits.csv                         # Train/Val/Test image index splits
    ├── model_details.json                 # Evaluated vision-language model architectures
    ├── prompt_templates.json              # Prompt engineering templates for zero-shot evaluation
    ├── zeroshot_probe_results.json        # Main benchmark results & accuracy tables
    ├── exact_duplicate_groups_all.csv     # Exact image duplicate clusters
    ├── near_duplicate_groups.csv          # Near-duplicate clusters identified via pHash/embeddings
    └── ... (28 total evaluation artifacts)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- `transformers`, `torchvision`, `scikit-learn`, `imagehash`, `pandas`, `numpy`

### Installation

```bash
git clone https://github.com/khushi-596/indian_food_vfm_benchmark.git
cd indian_food_vfm_benchmark
pip install -r requirements.txt
```

---

## 📊 Notebook Execution Pipeline

1. **`notebooks/food_research_01.ipynb`**: Preprocess dataset images, construct index tables, and generate reproducible train/val/test splits.
2. **`notebooks/food_research_02.ipynb`**: Run zero-shot classification across CLIP and vision backbones; train linear probes.
3. **`notebooks/food_research_04.ipynb`**: Benchmark model robustness against image corruptions and compute Grad-CAM heatmaps.
4. **`notebooks/food_research_06.ipynb`**: Conduct dataset deduplication audit using pHash and embedding similarity.
5. **`notebooks/food_research_07.ipynb`**: Execute multi-seed runs to compute confidence intervals and variance metrics.
6. **`notebooks/food_research_08.ipynb`**: Compute statistical significance tests (p-values, confidence bounds).
7. **`notebooks/food_research_09.ipynb`**: Compile summary tables and generate reproducibility artifacts.

---

## 📦 Reproducibility Package

The [`repro_package/`](./repro_package) directory contains all pre-computed metric outputs, prediction tables (`per_image_predictions.csv.gz`), split indices (`train_idx.pkl`, `val_idx.pkl`, `test_idx.pkl`), and deduplication summaries allowing full scientific replication without re-running long vision model evaluations.

---

## 📄 License & Attribution

This benchmark is released for academic and research purposes.
