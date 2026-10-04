# Hindi/Hinglish Sentiment Analysis

A deep learning project for **sentiment classification of Hindi/Hinglish review text** using LSTM-based architectures. The project implements and compares **Simple LSTM, Stacked BiLSTM, and CNN + BiLSTM** models for classifying text into sentiment categories.

## 📌 Project Overview

Hindi and Hinglish text often contains a mixture of Hindi and English words, making sentiment analysis challenging for traditional NLP models.

This project explores different deep learning architectures for sentiment classification and evaluates their performance using multiple metrics.

### Models Implemented

- **Simple LSTM** — Baseline recurrent neural network model
- **Stacked BiLSTM** — Multiple bidirectional LSTM layers for richer contextual representation
- **CNN + BiLSTM** — CNN for extracting local textual patterns followed by BiLSTM for sequential context modeling

The models are compared based on classification performance and computational complexity.

---

## 🎯 Objectives

- Perform sentiment analysis on Hindi/Hinglish text.
- Preprocess and tokenize review text for deep learning.
- Implement a **Simple LSTM** sentiment classifier.
- Implement a **Stacked BiLSTM** architecture.
- Implement a **CNN + BiLSTM** hybrid architecture.
- Compare the models using standard evaluation metrics.
- Analyze the trade-off between performance and model complexity.

---

## 🗂️ Project Structure

```text
hinglish-sentiment-model-comparison/
│
├── data/
│   └── hinglish_reviews.csv
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── models/
│   ├── simple_lstm.py
│   ├── stacked_bilstm.py
│   └── cnn_bilstm.py
│
├── results/
│   ├── architecture_comparison.csv   # mean ± std per architecture
│   ├── architecture_runs.csv         # one row per architecture × seed
│   ├── architecture_comparison.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── tokenizer.json
│   ├── label_classes.json
│   └── best_model.keras        # created by train.py (not tracked)
│
├── preprocessing.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔄 Methodology

The overall workflow is:

```text
Hindi/Hinglish Reviews
        ↓
Text Cleaning
        ↓
Tokenization
        ↓
Sequence Padding
        ↓
Word Embedding
        ↓
Deep Learning Model
        ↓
Sentiment Prediction
        ↓
Model Evaluation
```

### Preprocessing

The text data is processed before being provided to the neural networks. The preprocessing pipeline includes:

1. Removal of empty reviews
2. Text cleaning (`preprocessing.clean_text`): URLs and @mentions are removed, only Devanagari, Latin letters, and digits are kept, text is lowercased, and whitespace is collapsed
3. Deduplication **after cleaning**: reviews that become identical once cleaned are kept only once, and cleaned texts that carry conflicting labels are dropped. This guarantees no review appears in both the training and test sets (checked with an assertion in the code)
4. Stratified 70 / 15 / 15 split into training, validation, and test sets
5. Tokenization with a Keras `Tokenizer` (vocabulary of 20,000, `<OOV>` token) fitted on the training set only
6. **Pre-padding** (zeros added before the review) and truncation to 60 tokens. The median review is only 18 tokens, so a review typically gets about 42 padding steps. With pre-padding, the LSTM reads the real words last, so its final hidden state reflects the review rather than the padding. Only 3% of reviews are longer than 60 tokens and get cut
7. Balanced class weights to offset class imbalance

The same `clean_text` function is applied in `predict.py`, so new reviews are preprocessed exactly as the training data was.

---

## 🧠 Model Architectures

### 1. Simple LSTM

The Simple LSTM acts as the baseline model.

```text
Input Text (60 tokens)
    ↓
Embedding (128-d)
    ↓
SpatialDropout1D (0.3)
    ↓
LSTM (64 units, dropout 0.3, recurrent dropout 0.2)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (3, Softmax)
    ↓
Sentiment
```

LSTM is useful for capturing sequential dependencies in text while reducing the vanishing-gradient problems associated with traditional RNNs.

### 2. Stacked BiLSTM

The Stacked BiLSTM uses multiple bidirectional LSTM layers.

```text
Input Text (60 tokens)
    ↓
Embedding (128-d)
    ↓
SpatialDropout1D (0.3)
    ↓
BiLSTM (64 units per direction, return sequences, dropout 0.3, recurrent dropout 0.2)
    ↓
BiLSTM (32 units per direction, dropout 0.3, recurrent dropout 0.2)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (3, Softmax)
    ↓
Sentiment
```

The bidirectional architecture processes the sequence in both forward and backward directions, allowing the model to use contextual information from both sides of a word.

### 3. CNN + BiLSTM

The hybrid architecture combines convolutional feature extraction with bidirectional sequence modeling.

```text
Input Text (60 tokens)
    ↓
Embedding (128-d)
    ↓
SpatialDropout1D (0.3)
    ↓
Conv1D (64 filters, kernel 5, ReLU)
    ↓
MaxPooling1D (pool size 2)
    ↓
BiLSTM (64 units per direction, dropout 0.3, recurrent dropout 0.2)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (3, Softmax)
    ↓
Sentiment
```

The CNN extracts local patterns and important n-gram-like features, while the BiLSTM captures contextual and sequential relationships. MaxPooling halves the sequence length before the BiLSTM, which makes it cheaper to run.

All three models share the same embedding size, dropout settings, optimizer (Adam, learning rate 1e-3) and loss (sparse categorical cross-entropy), so the comparison isolates the effect of the architecture.

---

## 📊 Experimental Results

Test-set results from `results/architecture_comparison.csv`. Each architecture was trained **3 times (seeds 42, 123, 2024)** on the same fixed train/validation/test split (3,346 / 717 / 717 reviews), and the table shows **mean ± standard deviation** across the 3 runs. Results for each run are in `results/architecture_runs.csv`.

| Model | Accuracy | Macro F1 | Weighted F1 | RMSE | Parameters | Mean epochs |
|---|---:|---:|---:|---:|---:|---:|
| Simple LSTM | **96.61% ± 0.29** | **95.95% ± 0.32** | **96.62% ± 0.28** | 0.234 ± 0.010 | 610,819 | 10.3 |
| Stacked BiLSTM | 95.35% ± 1.03 | 94.51% ± 1.06 | 95.38% ± 1.02 | 0.269 ± 0.054 | 701,443 | 7.7 |
| CNN + BiLSTM | 96.47% ± 0.43 | 95.70% ± 0.38 | 96.47% ± 0.42 | **0.230 ± 0.032** | 670,531 | 8.7 |

Each model trains for up to 25 epochs with early stopping (patience 4 on validation loss), so the epoch counts differ.

**Findings**

- **All three architectures reach about 95–97% accuracy, and the differences between them are small.** Simple LSTM and CNN + BiLSTM are within one standard deviation of each other. Stacked BiLSTM is about 1–1.5 points lower and varies more between seeds.
- **Simple LSTM has the highest mean macro F1 and the smallest spread**, with the fewest parameters. Its best run (seed 2024: 96.93% accuracy, 0.963 macro F1) is saved as `results/best_model.keras`. CNN + BiLSTM has the lowest mean RMSE.
- **Why the simplest model is enough here:** the reviews are short (median 18 tokens) and sentiment is mostly carried by strong words such as *bakwas*, *zabardast* or *paisa barbaad*. Reading the review backwards or stacking more layers adds parameters without adding much useful information, and the larger Stacked BiLSTM overfits slightly sooner.
- **Padding matters more than architecture.** In an earlier run with post-padding (zeros after the review), the Simple LSTM got stuck near chance on 2 of 3 seeds (mean accuracy 70% ± 24) because its final state came after ~42 padding steps. Switching to pre-padding fixed this and raised its mean accuracy by 26 points; the BiLSTM models were barely affected because their backward LSTM reads the real words last either way. This was only visible because of the multi-seed evaluation.
- **Overfitting:** in `results/training_history.png`, training loss keeps falling towards 0 while validation loss levels off after about 3 epochs. Early stopping with `restore_best_weights=True` keeps the weights from the lowest validation loss.

### Hand-written test cases

The notebook (section 14) also tests the models on 21 new reviews written for this purpose, grouped into categories that are hard for sentiment models. Each architecture uses its best seed.

| Category | Review | Expected | Simple LSTM | Stacked BiLSTM | CNN + BiLSTM |
|---|---|---|---|---|---|
| Clear positive | product bahut accha hai, quality bhi badhiya hai | positive | ✓ | ✓ | ✓ |
| Clear positive | bohat zabardast cheez hai, seller ne time pe deliver kiya | positive | ✓ | ✓ | ✓ |
| Clear negative | bilkul bekaar product hai, paisa barbaad ho gaya | negative | ✓ | ✓ | ✓ |
| Clear negative | ghatiya quality hai, kabhi mat lena | negative | ✓ | ✓ | ✓ |
| Clear neutral | theek hai, price ke hisaab se average hai | neutral | ✓ | ✓ | ✓ |
| Clear neutral | product ok hai, na zyada acha na zyada bura | neutral | ✓ | ✓ | ✗ negative |
| English only | excellent product, very good quality, highly recommended | positive | ✓ | ✓ | ✓ |
| English only | worst product ever, totally waste of money | negative | ✓ | ✓ | ✓ |
| English only | it is okay, works as described, nothing special | neutral | ✗ positive | ✓ | ✓ |
| Negation | bilkul bhi acha nahi hai | negative | ✓ | ✓ | ✓ |
| Negation | koi problem nahi hai, sab sahi chal raha hai | positive | ✗ negative | ✗ negative | ✗ negative |
| Mixed opinion | quality achi hai lekin delivery bahut late thi | neutral | ✓ | ✓ | ✓ |
| Sarcasm | wah kya baat hai, do din mein hi toot gaya | negative | ✓ | ✓ | ✓ |
| Spelling variant | bht acha h yr, mza agya | positive | ✓ | ✓ | ✓ |
| Spelling variant | bkwas h bhai, pesy zaya | negative | ✓ | ✓ | ✓ |
| Very short | acha | positive | ✓ | ✓ | ✗ neutral |
| Very short | bekar | negative | ✓ | ✓ | ✓ |
| Caps / emoji | SUPERB!!! 😍😍 must buy | positive | ✓ | ✓ | ✓ |
| Domain: size | size chart galat hai, chhota aa gaya, return karna padega | negative | ✓ | ✓ | ✓ |
| Devanagari Hindi | यह प्रोडक्ट बहुत अच्छा है | positive | ✗ negative | ✗ neutral | ✗ neutral |
| Devanagari Hindi | बहुत खराब क्वालिटी है, पैसे बर्बाद | negative | ✓* | ✗ neutral | ✗ neutral |
| **Total** | | | **18 / 21** | **18 / 21** | **16 / 21** |

\* A lucky guess: none of the Devanagari words are in the vocabulary, so the model only sees unknown-word tokens.

What the cases show:

- **Strong points:** clear sentiment in Roman Hindi, Roman Urdu and English; spelling variants (*bht*, *bkwas*, *mza*); caps and emoji; mixed opinions; and the sarcasm case (the words *toot gaya* outweigh *wah kya baat hai*).
- **Double negation fails in every model.** In *"koi problem nahi hai"* the words *problem* and *nahi* each signal a complaint, and the models do not learn that together they mean "no problems". Simple negation (*"acha nahi hai"*) is handled correctly.
- **Devanagari script is not supported.** The dataset is entirely Roman-script, so every Devanagari word is unknown to the tokenizer and the predictions are guesses. Supporting Hindi script would need Devanagari training data or transliteration to Roman script before prediction.
- **Very short inputs** are less reliable for the CNN + BiLSTM: a single word gives the convolution little to work with.

---

## 📈 Evaluation Metrics

The models are evaluated using:

- **Accuracy** — Overall proportion of correctly classified samples.
- **Precision / Recall** — Per-class values, printed in the classification report during training.
- **Macro-F1** — Average F1-score across sentiment classes (used to rank the models).
- **Weighted-F1** — F1-score weighted according to class frequency.
- **RMSE** — Root Mean Squared Error between predicted and true label indices (negative = 0, neutral = 1, positive = 2). This is only a rough ordinal proxy.
- **Confusion Matrix** — Class-wise visualization of the best model's predictions.
- **Training curves** — Training vs validation loss and accuracy per epoch (`results/training_history.png`). A widening gap between training and validation loss shows overfitting.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sheetalambre-ai/hinglish-sentiment-model-comparison.git
cd hinglish-sentiment-model-comparison
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Place the dataset at `data/hinglish_reviews.csv`, with a `Review` column (text) and a `Sentiment` column (`negative`, `neutral`, or `positive`).

Train and compare all three architectures:

```bash
python train.py
```

This trains each architecture with 3 seeds and writes the per-run and mean ± std comparison tables, charts, best model, tokenizer, and label classes to `results/`.

Classify new reviews interactively with the best model:

```bash
python predict.py
```

The files in `models/` define the architectures only; they are imported by `train.py` and are not run on their own.

The full experiment is also in `notebooks/sentiment_analysis.ipynb`, which runs the same code as `train.py` step by step, with every cell's output saved (model summaries, per-epoch training logs, the results table, and all charts). Open it with `jupyter notebook`, or re-run it from the project root with:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/sentiment_analysis.ipynb
```

Training 3 architectures × 3 seeds takes roughly 15–20 minutes on a CPU. Section 14 of the notebook then runs all three models on the hand-written test cases.

---

## 📁 Dataset

**Source:** Zainab Ghaffar and Dr Muhammad Nouman Noor (2025). *Dataset for Sentiment Analysis in Code-Mixed Language*, Version 1. Mendeley Data. Published 10 October 2025. DOI: [10.17632/xxprggbztd.1](https://doi.org/10.17632/xxprggbztd.1)

The dataset contains user reviews collected from an e-commerce platform and written in low-resource code-mixed language: **Roman Urdu, Roman Hindi and Roman English**. Each review is labeled **Positive, Negative or Neutral**.

| | Count |
|---|---:|
| Reviews in the file | 5,034 |
| After cleaning and deduplication | 4,780 |
| Positive / Negative / Neutral (raw file) | 2,334 / 1,690 / 1,010 |

All reviews are in Latin (Roman) script; the dataset contains no Devanagari text. The cleaning step still keeps Devanagari characters, so Hindi-script input to `predict.py` is not stripped out.

> **Note:** `data/*.csv` is excluded from this repository by `.gitignore`. Download the dataset from the DOI link above and save it as `data/hinglish_reviews.csv`. Check the dataset page for its license before redistributing it.

---

## 🔬 Comparison

The project focuses on comparing increasingly sophisticated sequence architectures:

```text
Simple LSTM
     ↓
Stacked BiLSTM
     ↓
CNN + BiLSTM
```

This allows the effect of bidirectional processing, deeper recurrent layers, and convolutional feature extraction to be studied within the same sentiment-analysis task. In this experiment, once padding was handled correctly, the extra complexity gave no clear improvement over the Simple LSTM; see the findings under Experimental Results.

---

## 📌 Future Work

Possible extensions include:

- Transliterating Devanagari input to Roman script, or adding Devanagari training data
- More training examples of double negation (e.g. *"koi problem nahi"*)
- Transformer-based sentiment classification
- Multilingual BERT / IndicBERT
- Attention mechanisms
- Word2Vec or FastText embeddings
- Hyperparameter optimization
- Cross-dataset evaluation
- Handling spelling variations in Hinglish
- Comparison with traditional machine-learning approaches

---

## 📜 License

This project is intended for **academic and educational purposes**. Please check the original dataset's license and terms of use before redistributing the dataset.