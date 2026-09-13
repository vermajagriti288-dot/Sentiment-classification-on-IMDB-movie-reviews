# IMDB Sentiment Classification: RNN vs LSTM

A deep learning project comparing a **Simple Recurrent Neural Network (RNN)** and a **Long Short-Term Memory (LSTM)** model for binary sentiment classification of IMDB movie reviews.

## Objective

The project studies the working of LSTM networks and compares their performance with a traditional RNN on the same sentiment classification task.

## Tasks Covered

- Load and inspect the IMDB movie review dataset.
- Clean and preprocess review text.
- Encode positive and negative sentiment labels.
- Apply lowercasing, HTML/URL removal, punctuation and number removal, whitespace normalization, stopword removal, emoji removal, and stemming.
- Create TF-IDF features with 5,000 maximum features.
- Split the data into training, validation, and test sets.
- Build, train, and evaluate a Simple RNN.
- Build, train, and evaluate an LSTM.
- Measure accuracy, precision, recall, F1-score, confusion matrix, and training time.
- Produce a final RNN vs LSTM comparison table.

## Dataset

The notebook expects a local/Colab CSV file named:

`IMDB Dataset.csv`

The dataset contains movie reviews with binary sentiment labels (`positive` and `negative`).

> The dataset file is not included in this repository unless you have permission to redistribute it.

## Preprocessing

The notebook performs the following text preprocessing:

1. Lowercase conversion
2. HTML tag removal
3. URL removal
4. Punctuation removal
5. Number removal
6. Whitespace normalization
7. Stopword removal
8. Emoji removal
9. Porter stemming

The processed reviews are transformed into **TF-IDF features with up to 5,000 features**.

## Data Split

The notebook uses:

- 80% initial training / test split
- 20% of the training portion for validation
- `random_state=42`
- Stratification by sentiment label

## RNN Model

The Simple RNN uses:

- 2 recurrent layers
- 128 hidden units
- Dropout: 0.3
- Fully connected layers: 128 → 64 → 1
- ReLU activation
- Binary Cross-Entropy with Logits loss
- Adam optimizer
- Learning rate: 0.001
- Epochs: 10
- Batch size: 32

## LSTM Model

The LSTM uses:

- 2 LSTM layers
- 128 hidden units
- Dropout: 0.3
- Fully connected layers: 128 → 64 → 1
- ReLU activation
- Binary Cross-Entropy with Logits loss
- Adam optimizer
- Learning rate: 0.001
- Epochs: 10
- Batch size: 32

## Evaluation

The notebook evaluates both models using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Training time

The final comparison is generated in a Pandas DataFrame.

## Important Implementation Note

Although the project studies RNNs and LSTMs for sentiment classification, the uploaded notebook represents each review using a **5,000-dimensional TF-IDF vector** and feeds that vector as a **single sequence step** into the recurrent model.

This README intentionally documents the implementation as present in the submitted notebook rather than claiming a conventional word-token sequence LSTM architecture.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TensorFlow / Keras preprocessing utilities

## Repository Structure

```text
IMDB-Sentiment-RNN-vs-LSTM/
│
├── IMDB_Sentiment_RNN_vs_LSTM.ipynb
├── README.md
└── requirements.txt
```

## How to Run

### Google Colab

1. Upload `IMDB Dataset.csv` to the Colab environment.
2. Open the notebook in Google Colab.
3. Run the cells from top to bottom.

### Local Environment

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download the required NLTK resources when prompted by the notebook, then place `IMDB Dataset.csv` in the expected working directory.

## Conclusion

The project provides a practical comparison of Simple RNN and LSTM models for binary sentiment classification and records the requested evaluation metrics in a final comparison table.
