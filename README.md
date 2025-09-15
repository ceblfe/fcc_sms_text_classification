# # Neural Network SMS Text Classifier

Neural Network SMS Text Classifier.This project is the fifth project to get the Machine Learning with Python Certification from freeCodeCamp.

## Overview
This project implements a neural network using TensorFlow to classify SMS messages as either "ham" (normal messages) or "spam" (advertisements or unsolicited messages). The model is trained on the SMS Spam Collection dataset and integrated into a `predict_message` function that returns the probability of a message being spam and its corresponding label. The implementation is provided in the Jupyter notebook `fcc_sms_text_classification_CBF.ipynb`, designed to pass the FreeCodeCamp challenge by correctly classifying a set of test messages.

## Dataset
The dataset consists of TSV files from FreeCodeCamp:
- `train-data.tsv`: Training data with labeled SMS messages ("ham" or "spam").
- `valid-data.tsv`: Validation/test data with labeled SMS messages.

Each file contains two columns: the label ("ham" or "spam") and the message text. The dataset is imbalanced, with more "ham" messages than "spam," but the model handles this through binary classification.

## Requirements
To run the notebook, ensure you have the following dependencies:
- Python 3.x
- TensorFlow 2.16.1 (specified in the notebook for compatibility)
- Pandas
- NumPy
- Matplotlib
- tensorflow_datasets

The notebook includes installation commands for TensorFlow and tensorflow_datasets. Run it in an environment like Google Colab (with GPU support recommended for faster training) or Jupyter Notebook.

## How to Run
1. **Open the Notebook**: Load `fcc_sms_text_classification_CBF.ipynb` in Google Colab or Jupyter Notebook.
2. **Execute Cells in Order**:
   - The first few cells install and import libraries, including downgrading to TensorFlow 2.16.1 for compatibility.
   - Download the dataset files (`train-data.tsv` and `valid-data.tsv`).
   - Load and preprocess the data using Pandas.
   - Preprocess labels (ham=0, spam=1) and text (using `TextVectorization` for tokenization and sequencing).
   - Build the model with an embedding layer, global average pooling, dense layers, dropout, and sigmoid output.
   - Train the model for 10 epochs using the training dataset, with validation on the test dataset.
   - Define the `predict_message` function for inference.
   - Run the test cell to evaluate the model on predefined messages and check if the challenge is passed.
3. **Check Results**: The final cell will print "You passed the challenge. Great job!" if all test messages are classified correctly. Otherwise, it will indicate to keep trying.

## Model Details
- **Preprocessing**:
  - Labels: Mapped to binary values (ham=0, spam=1).
  - Text: Tokenized and padded to sequences of length 100 using `tf.keras.layers.TextVectorization` with a vocabulary size of 10,000.
  - Datasets: Batched TensorFlow datasets for efficient training.
- **Model Architecture**:
  - Embedding Layer: 10,000 vocabulary size, 128-dimensional embeddings, input length 100.
  - GlobalAveragePooling1D: Aggregates embeddings into a fixed-length vector.
  - Dense Layer: 64 units with ReLU activation and 50% dropout for regularization.
  - Output Layer: Single unit with sigmoid activation for spam probability.
- **Training**:
  - Optimizer: Adam.
  - Loss: Binary cross-entropy.
  - Metrics: Accuracy.
  - Epochs: 10 (sufficient for >95% validation accuracy).
- **Prediction Function**:
  - `predict_message(text)`: Vectorizes the input text, predicts the spam probability, and returns `[probability, label]` where label is "ham" if probability < 0.5, else "spam".

## Expected Output
After training, the model typically achieves ~98% accuracy on the validation set. The test cell evaluates on 7 sample messages and should output:
```
You passed the challenge. Great job!
```
If it fails, inspect the predictions for mismatches.

## Troubleshooting
- **NameError or Import Issues**: Ensure TensorFlow is version 2.16.1 and all imports are executed. Use `tf.keras.layers` explicitly for layers.
- **Low Accuracy/High MAE**: Increase epochs to 15, reduce dropout to 0.3, or adjust vocabulary size/sequence length (e.g., VOCAB_SIZE=5000, MAX_SEQUENCE_LENGTH=200).
- **Overfitting**: Monitor training vs. validation accuracy; add more dropout if needed.
- **Data Loading Errors**: Verify the TSV files download correctly and are tab-separated.
- **Environment**: If running locally, install dependencies via pip. In Colab, enable GPU for faster training.

## Notes
- The "spam" class is less frequent, but the model captures patterns like keywords ("sale," "win," "call") effectively.
- The notebook is self-contained and runs end-to-end in ~1-2 minutes on a GPU.
- Date of Implementation: Based on the current date (September 15, 2025), this solution uses TensorFlow 2.16.1 for stability.