
DistilBERT Sentiment Analysis on Amazon Reviews

This project demonstrates fine-tuning of a pre-trained DistilBERT model for sentiment analysis using Amazon Fine Food Reviews. The implementation uses the Hugging Face Transformers library and is optimized to run efficiently on Google Colab with GPU.

Model Summary

Model: DistilBERT (distilbert-base-uncased)
Task: Binary Sentiment Classification (Positive vs Negative)
Dataset: Amazon Fine Food Reviews (Kaggle)
Sample Size Used: 5000 reviews
Max Sequence Length: 128 tokens
Training Epochs: 2
Batch Size: 16 (train), 64 (eval)
Tokenizer: Hugging Face's DistilBertTokenizer

Libraries Used

transformers
scikit-learn
pandas
torch
Google Colab (with GPU)

Evaluation Results

Metric         Value
Accuracy       0.9275
Precision      0.9471
Recall         0.9670
F1 Score       0.9569
Training Time  87.06 sec
Testing Time   3.29 sec

Workflow

1. Load Amazon reviews dataset (5000 sample)
2. Preprocess: remove neutral reviews (Score = 3), create binary labels
3. Tokenize using DistilBERT tokenizer (max_length=128)
4. Create PyTorch-compatible dataset
5. Fine-tune DistilBertForSequenceClassification using Hugging Face Trainer
6. Evaluate on test set

File Structure

- DistilBERT_Sentiment_Analysis.ipynb: Jupyter notebook with full training pipeline
- Reviews.csv: Input dataset (Amazon reviews from Kaggle)
- README.md: Project summary and results

Future Work

- Compare with BERT and RoBERTa performance
- Add explainability using LIME or SHAP
- Apply zero-shot classification using GPT-2/3

License

This project is for educational and research purposes.
