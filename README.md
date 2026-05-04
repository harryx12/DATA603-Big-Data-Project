# Sentiment Analysis of Tweets using Big Data Platforms

A big data NLP pipeline that classifies tweet sentiment using the **Sentiment140** dataset (1.6M tweets). The project trains and compares five models — three Spark ML classifiers and two deep learning architectures — then deploys the best performer via an interactive Gradio web app.

---

## Overview

| Item | Detail |
|------|--------|
| **Dataset** | [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) — 1.6M labeled tweets |
| **Sample size** | 100,000 tweets (50K positive, 50K negative) |
| **Task** | Binary sentiment classification (Positive / Negative) |
| **Best model** | BiLSTM (AUC-ROC: 0.839, Avg Score: 0.798) |
| **Runtime** | Google Colab (GPU recommended) |

---

## Models Compared

| Model | Accuracy | AUC-ROC | Notes |
|-------|----------|---------|-------|
| Logistic Regression (Spark) | 0.720 | 0.773 | Strong, balanced baseline |
| Bernoulli Naive Bayes (Spark) | 0.742 | 0.524 | High accuracy, poor probability calibration |
| XGBoost (Spark) | 0.692 | 0.764 | Moderate; biased toward positive class |
| LSTM | 0.500 | 0.500 | Failed — predicted only one class |
| **BiLSTM** | **0.754** | **0.839** | **Best overall — most balanced** |

---

## Project Structure

```
notebook
├── Install Dependencies          # PySpark, XGBoost, Gradio, NLTK, etc.
├── Kaggle Credentials & Download # Pulls Sentiment140 via Kaggle API
├── Imports & NLTK Downloads
├── Create Spark Session
├── Load Dataset with PySpark
├── NLP Text Cleaning (UDF)       # Lowercase, URL/mention removal, lemmatization
├── Exploratory Data Analysis
│   ├── Class distribution (pie + bar)
│   ├── Text length distributions
│   ├── Top 20 unigrams per class
│   ├── Top 15 bigrams per class
│   └── Sentiment word clouds
├── Train / Test Split (80/20)
├── Spark ML Models
│   ├── Logistic Regression       # TF-IDF + LR pipeline
│   ├── Bernoulli Naive Bayes     # Binary HashingTF + NB
│   └── XGBoost (SparkXGBClassifier)
├── Deep Learning Models
│   ├── Keras Tokenizer & Sequence Prep
│   ├── LSTM
│   └── BiLSTM
├── Model Comparison & Visualization
│   ├── Training curves (loss & accuracy)
│   ├── Grouped bar chart (all models)
│   └── Confusion matrices (all 5 models)
├── Save Models & Best-Model Bundle (.pkl)
└── Gradio UI (standalone inference cell)
```

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or a local environment with Java installed for PySpark
- A Kaggle account with an API key

### Running the Notebook

1. **Open in Google Colab** and run cells sequentially from top to bottom.

2. **Set your Kaggle credentials** in the credentials cell:
   ```python
   KAGGLE_USERNAME = "your_username"
   KAGGLE_KEY      = "your_api_key"
   ```

3. **Install dependencies** (Cell 1 handles this automatically):
   ```
   pyspark==3.5.1, xgboost>=2.0.0, gradio, nltk, wordcloud, matplotlib, seaborn
   ```

4. **Run all training cells.** The notebook will automatically identify the best model by average accuracy + AUC-ROC score.

5. **Save the bundle.** A `best_model_bundle.pkl` file is generated containing the best model weights, tokenizer, and all evaluation results.

### Running the Gradio App (Standalone)

The final cell can be run independently in a **fresh session** without retraining:

1. Run only the last cell.
2. Upload `best_model_bundle.pkl` when prompted.
3. The Gradio interface will launch with a public share link.

---

## Text Preprocessing Pipeline

Raw tweets are cleaned via a Spark UDF backed by NLTK:

1. Lowercase
2. Remove URLs (`http://...`, `www...`)
3. Remove @mentions and #hashtags
4. Remove punctuation and non-alphabetic characters
5. Remove English stopwords
6. Lemmatize remaining tokens
7. Filter tokens shorter than 3 characters

---

## Key Visualizations Generated

| File | Description |
|------|-------------|
| `viz_01_class_distribution.png` | Pie + bar chart of class balance |
| `viz_02_text_length.png` | Word count histograms (raw vs. cleaned) |
| `viz_03_top_unigrams.png` | Top 20 words per sentiment class |
| `viz_04_top_bigrams.png` | Top 15 bigrams per sentiment class |
| `viz_07_wordclouds_combined.png` | Side-by-side sentiment word clouds |
| `viz_08_training_curves.png` | LSTM vs. BiLSTM loss & accuracy curves |
| `viz_09_model_comparison.png` | Grouped bar chart of all model scores |
| `viz_10_confusion_matrices.png` | Confusion matrices for all 5 models |

---

## Dependencies

```
pyspark==3.5.1
xgboost>=2.0.0
tensorflow
gradio
nltk
wordcloud
scikit-learn
matplotlib
seaborn
pandas
numpy
kaggle
joblib
```

---

## Notes

- The dataset is nearly perfectly balanced (50.1% negative / 49.9% positive), so accuracy is a reliable metric without needing class weighting.
- The vanilla LSTM failed to converge — its loss flatlined near 0.693 (log(2)), indicating it always predicted one class. The BiLSTM's bidirectional architecture resolved this.
- Bernoulli NB achieves the highest raw accuracy but its near-chance AUC-ROC (0.524) reveals poor probability calibration — it is not a reliable ranker.
- The saved `.pkl` bundle embeds Keras model weights as raw bytes, making it fully portable for the standalone Gradio cell.
