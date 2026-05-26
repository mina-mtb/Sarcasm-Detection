# Context-Aware and Retrieval-Augmented Sarcasm Detection

A research-oriented machine learning project for detecting sarcasm in Reddit conversations using classical NLP baselines, transformer fine-tuning, conversational context, retrieval-augmented features, and LLM-ready prompt construction.

This repository contains the implementation pipeline for the project **Context-Aware and Retrieval-Augmented Sarcasm Detection in Social Media**. The companion article repository is available here:

**Article repository:** [mina-mtb/Sarcasm_Detection_Article](https://github.com/mina-mtb/Sarcasm_Detection_Article)

---

## Overview

Sarcasm detection is difficult because sarcastic meaning often depends on the gap between a literal reply and the conversation that came before it. A short comment can look neutral on its own, but become sarcastic once the parent comment is included.

This project studies that problem through a staged experimental pipeline:

1. Build a reproducible dataset split and a TF-IDF baseline.
2. Fine-tune a context-free transformer on the target reply only.
3. Fine-tune a context-aware transformer on `parent_comment + comment`.
4. Add retrieval-neighbor features and LLM-style prompt examples for interpretability and future evaluation.

The final comparison shows that adding conversational context improves transformer performance, while retrieval-augmented stacking provides the strongest overall accuracy, balanced accuracy, macro-F1, and ROC-AUC in the saved run.

---

## Project Structure

```text
Sarcasm-Detection/
|-- data/
|   |-- train-balanced-sarcasm.csv
|   |-- train-balanced-sarc.csv.gz
|   |-- test-balanced.csv
|   `-- test-unbalanced.csv
|-- sarcasm_final_outputs/
|   |-- retrieval_metrics.json
|   |-- retrieval_knn_diagnostic_metrics.json
|   `-- llm_prompt_examples.txt
|-- 01_data_and_tfidf_baseline.ipynb
|-- 02_context_free_transformer.ipynb
|-- 03_context_aware_transformer.ipynb
|-- 04_retrieval_and_llm_prompts.ipynb
|-- .gitignore
`-- README.md
```

The notebooks are designed to be run in order. Each notebook saves outputs that are reused by the next stage.

---

## Repository Ecosystem

This project is split into two connected repositories:

| Repository | Purpose |
|---|---|
| This repository | Code, experiments, notebooks, metrics, retrieval diagnostics, and prompt examples |
| [Sarcasm_Detection_Article](https://github.com/mina-mtb/Sarcasm_Detection_Article) | Research article, paper source files, bibliography, and written analysis |

Use this repository to reproduce the experiments, and use the article repository for the formal research write-up and academic context.

---

## Dataset

The project uses the Reddit sarcasm dataset available on Kaggle under the notebook configuration:

```python
DATASET_SLUG = "/kaggle/input/datasets/danofer/sarcasm"
```

Expected fields include:

| Column | Description |
|---|---|
| `label` | Binary sarcasm label, where `1` means sarcastic and `0` means non-sarcastic |
| `comment` | Target Reddit reply to classify |
| `parent_comment` | Previous conversational context |
| `subreddit` | Subreddit source |
| `score`, `ups`, `downs` | Reddit interaction metadata |
| `author`, `date`, `created_utc` | User and time metadata |

The main experiments use balanced train/test data so that accuracy, balanced accuracy, F1, and macro-F1 can be compared reliably.

> Note: Large dataset and model artifacts are intentionally ignored by Git through `.gitignore`.

---

## Methodology

### 1. Data Preparation and TF-IDF Baseline

Notebook: `01_data_and_tfidf_baseline.ipynb`

This notebook creates the shared experimental foundation:

- Loads and standardizes the Reddit sarcasm dataset.
- Cleans text fields and prepares `comment` and `parent_comment`.
- Creates train, validation, and test splits.
- Trains a context-free TF-IDF + Logistic Regression baseline.
- Saves baseline predictions, metrics, plots, and reusable split files.

The TF-IDF model is intentionally simple. It provides a transparent baseline for measuring the value of transformer representations and conversational context.

### 2. Context-Free Transformer

Notebook: `02_context_free_transformer.ipynb`

This stage fine-tunes `distilroberta-base` using only the target reply:

```text
Input = comment
```

The goal is to test how well a transformer can detect sarcasm from the reply alone, without explicit conversational context.

Main configuration:

| Parameter | Value |
|---|---|
| Base model | `distilroberta-base` |
| Max sequence length | `256` |
| Batch size | `8` |
| Epochs | `3` |
| Device | CUDA when available, otherwise CPU |

### 3. Context-Aware Transformer

Notebook: `03_context_aware_transformer.ipynb`

This stage fine-tunes the same transformer architecture, but includes the parent comment:

```text
Input = parent_comment + comment
```

This is the main context-aware model. It tests the central hypothesis of the project: sarcasm detection improves when the model can compare a reply against the conversation that triggered it.

The notebook also performs error analysis by comparing model behavior across TF-IDF, context-free transformer, and context-aware transformer predictions.

### 4. Retrieval-Augmented Evaluation and LLM Prompts

Notebook: `04_retrieval_and_llm_prompts.ipynb`

This final notebook extends the supervised models with retrieval-based information:

- Builds dense sentence embeddings with `BAAI/bge-base-en-v1.5`.
- Uses FAISS for nearest-neighbor retrieval.
- Evaluates a retrieval-only kNN diagnostic baseline.
- Trains a calibrated logistic stacker using supervised probabilities plus retrieval-neighbor features.
- Saves retrieval metrics and prompt examples.
- Generates zero-shot and few-shot prompt templates for future LLM-based sarcasm classification.

The retrieval-only diagnostic is not expected to outperform fine-tuned transformers, because semantic similarity does not always imply the same sarcasm label. Its main role is to support analysis, retrieval features, and prompt construction.

---

## Results

Saved full-test metrics from the current run:

| Model | Input | Accuracy | Balanced Acc. | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | `comment` | 0.7220 | 0.7220 | 0.7392 | 0.6862 | 0.7117 | 0.7217 | 0.7948 |
| Context-free Transformer | `comment` | 0.7725 | 0.7725 | 0.7569 | 0.8029 | 0.7792 | 0.7723 | 0.8585 |
| Context-aware Transformer | `parent_comment + comment` | 0.7824 | 0.7824 | 0.7648 | 0.8159 | 0.7895 | 0.7822 | 0.8694 |
| Retrieval-augmented calibrated stacker | supervised probabilities + retrieval features | 0.7902 | 0.7902 | 0.8093 | 0.7594 | 0.7835 | 0.7900 | 0.8705 |

Additional retrieval diagnostic:

| Model | Accuracy | Balanced Acc. | F1 | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Retrieval-only kNN diagnostic, `k=5` | 0.5795 | 0.5795 | 0.5957 | 0.5788 | 0.6086 |

Key observations:

- The context-free transformer strongly improves over the TF-IDF baseline.
- The context-aware transformer improves F1 by approximately `+0.0103` over the context-free transformer.
- The retrieval-augmented calibrated stacker achieves the best accuracy, balanced accuracy, precision, macro-F1, weighted-F1, and ROC-AUC in the saved run.
- The context-aware transformer achieves the best recall and best average precision among the reported supervised models.

---

## Final Saved Outputs

The repository currently includes these final output artifacts:

| File | Description |
|---|---|
| `sarcasm_final_outputs/retrieval_metrics.json` | Full-test retrieval-augmented calibrated stacker metrics |
| `sarcasm_final_outputs/retrieval_knn_diagnostic_metrics.json` | Retrieval-only kNN diagnostic metrics |
| `sarcasm_final_outputs/llm_prompt_examples.txt` | Zero-shot and few-shot LLM prompt examples generated from retrieved neighbors |

Intermediate notebook outputs such as model checkpoints, CSV predictions, FAISS indexes, plots, and zipped Kaggle outputs are excluded from Git when they are large or reproducible.

---

## Installation

Create a Python environment and install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib tqdm
pip install torch transformers accelerate datasets
pip install sentence-transformers faiss-cpu
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation page.

Optional but recommended:

```bash
pip install jupyter ipykernel
```

---

## How to Run

Run the notebooks in this order:

```text
01_data_and_tfidf_baseline.ipynb
02_context_free_transformer.ipynb
03_context_aware_transformer.ipynb
04_retrieval_and_llm_prompts.ipynb
```

Recommended execution environment:

| Component | Recommendation |
|---|---|
| Runtime | Kaggle Notebook or local Jupyter environment |
| GPU | NVIDIA GPU for transformer fine-tuning |
| Memory | High RAM recommended for full dataset and retrieval stages |
| Storage | Several GB for datasets, checkpoints, predictions, and indexes |

The notebooks are configured to run on Kaggle-style paths, but the same workflow can be adapted locally by placing the dataset files in `data/` and updating the input/output paths where needed.

---

## Reproducibility Notes

- All main notebooks use the same shared split generated in Notebook 1.
- Transformer stages reuse the split files and comparison tables from previous notebooks.
- Retrieval evaluation uses the already generated supervised predictions.
- The final stacker uses calibrated logistic regression over supervised probabilities and retrieval-neighbor features.
- The full reported test set contains `151,496` samples with approximately balanced positive and negative classes.

Because transformer fine-tuning can vary slightly across hardware, driver versions, and random seeds, small metric differences may occur when rerunning the full pipeline.

---

## LLM Prompting Component

The project includes prompt construction for LLM-based sarcasm classification. The saved prompt examples support:

- Zero-shot classification using only the parent comment and target reply.
- Few-shot classification using retrieved similar labeled examples.
- Short explanation requests alongside binary labels.

The prompt format uses:

```text
Use 1 for sarcastic and 0 for non-sarcastic.
```

No final LLM accuracy is reported in this repository because no `llm_predictions.csv` file is included in the saved outputs. The prompt examples are prepared for future LLM evaluation.

---

## Research Questions

This implementation supports the following research questions:

1. How much does transformer fine-tuning improve sarcasm detection over a TF-IDF baseline?
2. Does adding conversational context improve sarcasm classification?
3. Can retrieval-neighbor features improve supervised model predictions?
4. Can retrieved examples provide useful few-shot demonstrations for LLM-based classification?

---

## Citation and Article

For the full written article, methodology discussion, and academic references, see:

[mina-mtb/Sarcasm_Detection_Article](https://github.com/mina-mtb/Sarcasm_Detection_Article)

Suggested project reference:

```text
Tahmasebi, M. Context-Aware and Retrieval-Augmented Sarcasm Detection in Social Media.
Code repository: Sarcasm-Detection.
Article repository: https://github.com/mina-mtb/Sarcasm_Detection_Article
```

---

## Project Status

This repository contains a complete experimental notebook pipeline with saved final retrieval metrics and LLM prompt examples. Future extensions may include:

- Running and reporting LLM classification accuracy.
- Packaging the notebooks into reusable Python modules.
- Adding automated experiment tracking.
- Publishing trained model artifacts separately from the Git repository.

---

## Acknowledgment

Developed as part of the **Advanced Machine Learning with Neural Networks** coursework.
