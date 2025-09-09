# VoiceScope
# ABSTRACT
VoiceScope, aims to explore and compare how well two deep learning architectures perform:
Long Short-Term Memory (LSTM) and a hybrid Convolutional Neural Network with
Bidirectional LSTM (CNN-BiLSTM). These models are selected for their ability to capture
temporal patterns and spatial hierarchies in sequential data like speech. Voice inputs go through
a preprocessing pipeline that extracts features such as pitch, magnitude, Mel-frequency cepstral
coefficients (MFCCs), and filter-bank energies, which are commonly used in speech signal
analysis. Both models are then trained independently using these features. The main focus is a
comparative study of the two architectures regarding classification accuracy, training time,
complexity, and their ability to generalize to unseen data. The project's outcome demonstrate
which model performs better than the other in real-world voice-based demographic
classification.

Keywords: LSTM, CNN-BiLSTM, voice classification, pitch, MFCC, filter-bank, age and
gender recognition, deep learning, model comparison.
# OBJECTIVES
To classify speaker gender (male/female) and age groups (20s–60s) from voice signals.

To preprocess audio data and extract relevant spectral and temporal features.

To design and implement LSTM and CNN–BiLSTM models for demographic classification.

To perform a comparative analysis of model performance using accuracy, precision, recall, and F1-score.
# SCOPE
Focuses only on age and gender classification from speech.

Uses deep learning techniques (LSTM and CNN–BiLSTM).

Considers filtered subsets of publicly available voice datasets.

Provides insights into the strengths and limitations of temporal vs. hybrid models.
# METHODOLOGY
1. Data Collection & Filtering

    Gather publicly available speech dataset.
    
    Filter samples with valid age (20s–60s) and gender (male/female) labels.

2. Preprocessing

    Convert audio files into a consistent format (mono, fixed sampling rate).
    
    Extract MFCCs and spectrograms as input features.
    
    Normalize features and split data into training, validation, and test sets.

3. Model Development

    LSTM Model: Designed to capture temporal dependencies in MFCC sequences.
    
    CNN–BiLSTM Model: CNN layers extract spectral patterns from spectrograms, followed by BiLSTM for temporal modeling.

4. Training & Evaluation

    Train both models on the prepared datasets.
    
    Evaluate using metrics: accuracy, precision, recall, F1-score, confusion matrix.

5. Comparative Analysis

    Compare the performance of LSTM vs. CNN–BiLSTM.
    
    Highlight strengths, weaknesses, and potential applications.
    
