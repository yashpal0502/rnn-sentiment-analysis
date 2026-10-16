# RNN Sentiment Analysis

Binary sentiment classification of movie reviews (**Positive / Negative**) using a sequence-based Simple RNN built from scratch in PyTorch — an end-to-end pipeline from raw text to a saved model and real-review inference.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-RNN-ee4c2c)
![Task](https://img.shields.io/badge/task-binary%20sentiment-informational)
![Status](https://img.shields.io/badge/status-baseline-yellow)

> **Status:** this repository is the **baseline** for a planned LSTM / GRU / Bidirectional comparison. The Simple RNN is intentionally untuned. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## Table of Contents

- [Highlights](#highlights)
- [Results](#results)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model & Hyperparameters](#model--hyperparameters)
- [Example Predictions](#example-predictions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Author](#author)

---

## Highlights

- **Leak-free pipeline** — stratified 80/20 split, and the vocabulary is built from training reviews only.
- **Negation-aware preprocessing** — `not`, `no`, `nor`, `never`, `neither` and `n't` are kept when stopwords are removed.
- **Proper variable-length handling** — padded batches are fed through `pack_padded_sequence`, so padding is never treated as real input.
- **Numerically stable loss** — `BCEWithLogitsLoss` on a single logit.
- **Save → reload → predict** — the trained weights are persisted and reused in a separate inference notebook.

## Results

Evaluated on a held-out test set of **9,917 reviews** (never used for vocabulary building).

| Metric | Score |
|---|---:|
| Accuracy | **76.21%** (95% CI ≈ ±0.84 pp) |
| Precision (positive class) | 0.7405 |
| Recall (positive class) | 0.8097 |
| F1 (positive class) | 0.7736 |
| Macro-F1 | 0.7615 |

<p align="center">
  <img src="assets/confusion_matrix.png" width="46%" alt="Confusion matrix on the test set">
  <img src="assets/training_loss.png" width="52%" alt="Training loss per epoch">
</p>

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| **Actual Negative** | TN = 3,528 | FP = 1,412 |
| **Actual Positive** | FN = 947 | TP = 4,030 |

**Reading the numbers:** the model finds 81.0% of positive reviews but only 71.4% of negative ones, i.e. it is biased toward predicting *Positive*. Details are in [REPORT.md](REPORT.md).

**Baseline comparison:** an earlier TF-IDF → single-time-step RNN reached ≈ **84.21%** accuracy. That is a *different pipeline* (no embeddings, no real sequence), so the two numbers are not an apples-to-apples architecture comparison — but it does show the current sequence RNN has clear room to improve.

## How It Works

```mermaid
flowchart LR
    A[Raw reviews] --> B[Text preprocessing]
    B --> C[Stratified 80/20 split]
    C --> D[Train-only vocabulary]
    D --> E[Token IDs]
    E --> F[Pad / truncate to 100]
    F --> G[Embedding 128-d]
    G --> H[Simple RNN 128 units]
    H --> I[Linear → 1 logit]
    I --> J[Sigmoid]
    J --> K[Positive / Negative]
```

**Model math.** The RNN updates its hidden state at every time step:

$$h_t = \tanh\left(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h\right)$$

The final hidden state is mapped to a logit and squashed into a probability:

$$z = W h_T + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Decision rule: `p ≥ 0.5 → Positive`, otherwise `Negative`. Training minimises binary cross-entropy:

$$L = -\big[\,y \log p + (1-y)\log(1-p)\,\big]$$

## Project Structure

```text
RNN/
├── assets/                      
│   ├── confusion_matrix.png
│   └── training_loss.png
├── data/
│   ├── IMDB_Dataset.csv          
│   ├── processed_reviews.csv    
│   └── rnn_data.pt              
├── models/
│   └── sentiment_rnn.pth        
├── src/
│   ├── 01_preprocessing.ipynb
│   ├── 02_dataset_dataloader.ipynb
│   ├── 03_build_rnn.ipynb
│   ├── 04_rnn_preparation.ipynb
│   ├── 05_rnn_model.ipynb
│   └── 06_prediction.ipynb
├── .gitignore
├── LICENSE
├── README.md
├── REPORT.md                    
└── requirements.txt
```

### Notebook guide

| Notebook | Purpose |
|---|---|
| `01_preprocessing.ipynb` | Lowercase, strip URLs / HTML / punctuation, tokenize, remove stopwords (keeping negations), Porter-stem → `processed_reviews.csv` |
| `02_dataset_dataloader.ipynb` | *Earlier stage* — TF-IDF features and DataLoaders (used for the 84.21% baseline) |
| `03_build_rnn.ipynb` | *Earlier stage* — first RNN on TF-IDF input |
| `04_rnn_preparation.ipynb` | Train/test split, train-only vocabulary (`<PAD>=0`, `<UNK>=1`), integer encoding, padding/truncation to 100, length tracking, tensors and DataLoaders → `rnn_data.pt` |
| `05_rnn_model.ipynb` | **Final baseline:** Embedding(128) → Simple RNN(128) → Linear(1); training, evaluation, model saving |
| `06_prediction.ipynb` | Loads the saved model and vocabulary and classifies new reviews |

## Getting Started

### Prerequisites

- Python 3.9+
- A CPU is enough (a GPU is optional)

### Installation

```bash
git clone https://github.com/yashpal0502/rnn-sentiment-analysis.git

python -m venv .venv
source .venv/bin/activate        

pip install -r requirements.txt
```

NLTK needs its data files once (stopwords and tokenizer):

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
```

## Usage

### Run the full pipeline

Execute the notebooks **in order** from the `src/` folder:

```text
01_preprocessing.ipynb
04_rnn_preparation.ipynb
05_rnn_model.ipynb
06_prediction.ipynb
```

`02` and `03` document the earlier TF-IDF experiment and are **not required** to reproduce the current baseline.

### Predict on your own reviews

1. Make sure `data/rnn_data.pt` and `models/sentiment_rnn.pth` exist (created by notebooks 04 and 05).
2. Open `src/06_prediction.ipynb` and enter your review text.
3. The notebook applies the **same preprocessing and training vocabulary**, then prints the sentiment and the positive-class probability.

> **Loading note:** only the `state_dict` is saved. Recreate the exact same model architecture (same vocabulary size, embedding dim, hidden size) before calling `load_state_dict`.

## Model & Hyperparameters

| Component | Setting |
|---|---|
| Dataset size | 49,582 reviews |
| Split | 80/20 stratified, `random_state=42` → 39,665 train / 9,917 test |
| Vocabulary | 124,455 tokens (training data only) |
| Max sequence length | 100 |
| Embedding | 128-d, `padding_idx=0` |
| Recurrent layer | Simple RNN (tanh), 1 layer, 128 hidden units |
| Sequence handling | `pack_padded_sequence(..., enforce_sorted=False)` |
| Output | `Linear(128 → 1)` logit |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam, learning rate 0.001 |
| Batch size / epochs | 64 / 10 |
| Trainable parameters | ≈ 15.96 M (≈ 99.8% in the embedding table) |

## Example Predictions

The probability is the model's estimate that *that review* is positive — **not** the model's accuracy.

| Review | Prediction | P(positive) |
|---|---|---:|
| Amazing acting and a great story. | Positive | 0.9850 |
| This movie was absolutely amazing. I really loved it! | Positive | 0.9427 |
| This movie was fantastic. I loved every minute of it! | Positive | 0.9218 |
| I did not enjoy this movie at all. | Negative | 0.1886 |
| I really hated this movie. | Negative | 0.1356 |
| This was the worst movie I have ever watched. | Negative | 0.0258 |
| The movie was okay, nothing special. | Negative | 0.0124 |

The last row shows the binary limitation: a lukewarm review is forced into one of two classes, and the model is very confident about it.

## Limitations

- Simple RNNs struggle with long-range dependencies (vanishing gradients).
- No validation set — there is no clean data for early stopping or tuning.
- Large vocabulary (124k) relative to ~40k training reviews → many rare, poorly learned embeddings.
- Fixed 100-token window; longer reviews are truncated.
- Binary labels only; no neutral class.
- Single run, no seed sweep, no systematic hyperparameter search.
- Classic preprocessing (stemming, stopword removal) is not necessarily optimal for neural models.

## Roadmap

- [ ] Add a validation split (train / val / test)
- [ ] Hyperparameter tuning (hidden size, embedding size, learning rate, sequence length, vocabulary cut-off)
- [ ] Dropout and other regularisation
- [ ] Bidirectional RNN
- [ ] LSTM and GRU, compared under the same protocol
- [ ] Pre-trained embeddings (GloVe / fastText)
- [ ] Error analysis on misclassified reviews
- [ ] Multiple seeds with mean ± std
- [ ] Standalone `predict.py` / small API

## Reproducibility

- Data split: `random_state=42`, `stratify=labels`.
- Vocabulary is built **only** from the training split.
- Record your own environment for any reported run — Python / PyTorch version, hardware, training time, and the PyTorch random seed (`torch.manual_seed(...)`) — see the [reproducibility checklist](REPORT.md#a3-reproducibility-checklist) in the report.

## Documentation

- 📄 **[REPORT.md](REPORT.md)** — full technical report: methodology, experiments, results, error discussion, limitations.

## Author

Yashpal