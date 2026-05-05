# RAM-SD: Context-Aware and Retrieval-Augmented Sarcasm Detection

This project implements a state-of-the-art sarcasm detection pipeline based on the research paper: **"Context-Aware and Retrieval-Augmented Sarcasm Detection in Social Media"**.

The core idea is to enhance standard transformer-based classifiers (like RoBERTa) by retrieving similar sarcastic examples from a large-scale knowledge base and using them as additional context for the model.

## 🚀 Pipeline Architecture

The project is divided into four main phases, each corresponding to a Jupyter notebook:

### 1. Preprocessing (`preprocessing.ipynb`)
- **Objective**: Clean and prepare the Reddit Sarcasm dataset.
- **Key Steps**:
    - Text cleaning (URL removal, whitespace normalization).
    - **Context Merging**: Combining the `parent_comment` and `comment` into a single field to provide situational context.
    - Basic EDA on class distribution and comment lengths.

### 2. Knowledge Base Creation (`2a_knowledge_base.ipynb`)
- **Objective**: Build a searchable index of sarcastic expressions.
- **Key Steps**:
    - Filter the dataset for sarcastic examples only (`label == 1`).
    - Use `sentence-transformers/all-MiniLM-L6-v2` to generate dense vector embeddings.
    - Build a **FAISS Index** for high-speed similarity search.

### 3. Retrieval Augmentation (`2b_Retrieval.ipynb`)
- **Objective**: Augment the training/test samples with retrieved context.
- **Key Steps**:
    - For every comment in the dataset, retrieve the Top-K (default K=3) most similar sarcastic examples from the FAISS Knowledge Base.
    - Append these examples to the original text using a `[SIMILAR]` delimiter.
    - Create a new `augmented_text` column.

### 4. RoBERTa Classification (`2c_Roberta.ipynb`)
- **Objective**: Fine-tune a classifier on the augmented data.
- **Key Steps**:
    - Load `roberta-base`.
    - Tokenize the augmented text (concatenation of current context + retrieved similar cases).
    - Train with **Mixed Precision (FP16)** on Kaggle T4 GPU.
    - Evaluate using Accuracy and F1-score.

## 🛠️ Requirements

To run this project, you need the following libraries:

```bash
pip install pandas numpy matplotlib seaborn faiss-cpu sentence-transformers datasets transformers accelerate scikit-learn tqdm
```

## 📊 Dataset
The project uses the **SARC (Self-Annotated Reddit Corpus)** dataset, specifically the balanced version. 
- **Source**: Kaggle / Reddit
- **Columns**: `label`, `comment`, `author`, `subreddit`, `score`, `ups`, `downs`, `date`, `created_utc`, `parent_comment`.

## 📖 Reference
> Tahmasebi, M. (2026). Context-Aware and Retrieval-Augmented Sarcasm Detection in Social Media.

---
*Developed for the Advanced Machine Learning with Neural Networks course.*
