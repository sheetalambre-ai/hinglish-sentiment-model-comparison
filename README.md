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
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── training_history.png
│
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

1. Text cleaning
2. Tokenization
3. Conversion of text into numerical sequences
4. Padding/truncation of sequences
5. Preparation of training, validation, and test sets

---

## 🧠 Model Architectures

### 1. Simple LSTM

The Simple LSTM acts as the baseline model.

```text
Input Text
    ↓
Embedding
    ↓
LSTM
    ↓
Dense
    ↓
Sentiment
```

LSTM is useful for capturing sequential dependencies in text while reducing the vanishing-gradient problems associated with traditional RNNs.

### 2. Stacked BiLSTM

The Stacked BiLSTM uses multiple bidirectional LSTM layers.

```text
Input Text
    ↓
Embedding
    ↓
BiLSTM
    ↓
BiLSTM
    ↓
Dense
    ↓
Sentiment
```

The bidirectional architecture processes the sequence in both forward and backward directions, allowing the model to use contextual information from both sides of a word.

### 3. CNN + BiLSTM

The hybrid architecture combines convolutional feature extraction with bidirectional sequence modeling.

```text
Input Text
    ↓
Embedding
    ↓
CNN
    ↓
BiLSTM
    ↓
Dense
    ↓
Sentiment
```

The CNN extracts local patterns and important n-gram-like features, while the BiLSTM captures contextual and sequential relationships.

---

## 📊 Experimental Results

The following results were obtained from the implemented architectures:

| Model | Accuracy | Macro F1 | Weighted F1 | RMSE | Parameters | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| Simple LSTM | 93.61% | 92.40% | 93.67% | 0.2981 | 614,659 | 25 |
| Stacked BiLSTM | 93.61% | 92.81% | 93.58% | 0.3613 | 705,283 | 10 |
| CNN + BiLSTM | **94.86%** | **94.04%** | **94.89%** | **0.2911** | 674,371 | 8 |

The experiments show that all three architectures achieve strong sentiment-classification performance, while the CNN + BiLSTM architecture achieves the highest accuracy and Macro-F1 among the evaluated models.

---

## 📈 Evaluation Metrics

The models are evaluated using:

- **Accuracy** — Overall proportion of correctly classified samples.
- **Precision** — Proportion of predicted positive instances that are correct.
- **Recall** — Proportion of actual positive instances correctly identified.
- **Macro-F1** — Average F1-score across sentiment classes.
- **Weighted-F1** — F1-score weighted according to class frequency.
- **RMSE** — Root Mean Squared Error between predicted and target representations.
- **Confusion Matrix** — Class-wise visualization of prediction performance.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/hinglish-sentiment-model-comparison.git
cd hinglish-sentiment-model-comparison
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Run the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/sentiment_analysis.ipynb
```

Alternatively, run the model scripts individually from the `models/` directory.

---

## 📁 Dataset

The project uses a labeled Hindi/Hinglish sentiment dataset containing review text and corresponding sentiment labels.

The dataset is used for supervised sentiment classification.

> **Note:** Dataset files may be excluded from this repository if their redistribution is restricted. Please refer to the original dataset source for downloading the data.

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

This allows the effect of bidirectional processing, deeper recurrent layers, and convolutional feature extraction to be studied within the same sentiment-analysis task.

---

## 📌 Future Work

Possible extensions include:

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