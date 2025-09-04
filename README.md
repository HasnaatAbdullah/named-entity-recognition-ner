# named-entity-recognition-ner
Implementation of a Named Entity Recognition (NER) system using TensorFlow and BiLSTMs, capable of identifying entities like persons, locations, organizations, and dates.
# 🏷️ Named Entity Recognition (NER) using TensorFlow & BiLSTM

This project implements a **Named Entity Recognition (NER)** model using **TensorFlow** and **BiLSTM** networks.  
NER is a key task in **Natural Language Processing (NLP)** that involves identifying and classifying entities in text such as **persons, organizations, locations, dates**, and more.

The project demonstrates **data preprocessing**, **label encoding**, **sequence padding**, **LSTM-based architecture design**, **masked loss computation**, and **model evaluation**.

---

## 🚀 Project Overview

**Named Entity Recognition (NER)** is the process of detecting and classifying named entities within unstructured text.  
For example, given the sentence:

> "French President Emmanuel Macron visited Morocco on Christmas."

The model predicts:

| Token      | Label       |
|-----------|------------|
| French    | B-GPE      |
| President | O          |
| Emmanuel  | B-PER      |
| Macron    | I-PER      |
| visited   | O          |
| Morocco   | B-GPE      |
| on        | O          |
| Christmas | B-DATE     |

Where:
- **B-GPE** → Beginning of a geopolitical entity
- **B-PER** → Beginning of a person’s name
- **B-DATE** → Date entity
- **O** → Not an entity

---

## 🧠 Key Features

### **1. Data Preprocessing**
- Tokenizes sentences into words.
- Encodes text and labels into numerical representations.
- Applies **padding** to handle variable sequence lengths.
- Builds a **label vectorizer** for mapping entity tags.

### **2. Model Architecture**
- Embedding Layer for word representation.
- **Bi-directional LSTM** for contextual understanding.
- Time-distributed Dense Layer to predict entity tags for each token.
- Masking to handle padded sequences.
- Uses **Sparse Categorical Cross-Entropy Loss** with masking.

### **3. Training & Evaluation**
- Trains on labeled NER data.
- Evaluates using **accuracy** and **F1-score**.
- Achieves high performance on both training and test sets.

### **4. Inference on Custom Sentences**
- Pass any sentence to the trained model.
- Returns predicted entities along with their tags.

---

## 🏗️ Model Architecture

- **Input Layer:** Encoded token IDs
- **Embedding Layer:** Learns dense word embeddings
- **BiLSTM Layer:** Captures context from both directions
- **Dense Layer:** Outputs probabilities for each possible tag
- **Output Layer:** Entity classification per token

---

## 📌 Implementation Details

### **1. Encoding Sentences & Labels**
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(ner_labels)

padded_sequences = pad_sequences(encoded_labels, maxlen=MAX_LEN, padding="post")

