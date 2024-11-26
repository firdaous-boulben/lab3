# Lab 3

## Objective
The primary goal of this lab is to explore the application of deep learning models in the field of Natural Language Processing (NLP). By leveraging sequence-based models such as Recurrent Neural Networks (RNN), Bidirectional RNNs, GRUs, and LSTMs, we aim to gain hands-on experience in training models for text classification tasks. Additionally, this lab will introduce the use of pre-trained transformer models, specifically GPT-2, for text generation. The lab is structured into two main parts:

### Part 1: Classification Task
- Collect text data in Arabic from multiple websites using web scraping tools such as Scrapy or BeautifulSoup.
- Preprocess the data by performing tokenization, stemming, lemmatization, stop word removal, and discretization.
- Train several deep learning models (RNN, Bidirectional RNN, GRU, LSTM) for text classification and tune hyperparameters for the best performance.
- Evaluate the models using standard metrics (accuracy, precision, recall, F1-score) and additional metrics like BLEU score.

### Part 2: Transformer for Text Generation
- Install and fine-tune the GPT-2 pre-trained model using a custom [dataset](https://www.kaggle.com/datasets/adarshpathak/shakespeare-text).
- Use the fine-tuned model to generate new text based on a given sentence.

## Key Takeaways
During this lab, I learned how to:
1. **Preprocess Text Data:** I gained experience in handling Arabic text data by applying various NLP preprocessing techniques like tokenization, stemming, lemmatization, and stop word removal.
2. **Train Sequence Models:** I implemented and trained multiple sequence models (RNN, Bi-RNN, GRU, LSTM) for text classification tasks using PyTorch.
3. **Evaluate Model Performance:** I learned how to evaluate models using standard classification metrics like accuracy, precision, recall, F1-score, as well as additional metrics like BLEU score.
4. **Fine-Tune Transformers:** I explored how to fine-tune the GPT-2 pre-trained model for text generation and observed how pre-trained models can be adapted to custom tasks.
5. **Hyperparameter Tuning:** I gained practical experience in tuning hyperparameters to improve model performance and minimize overfitting or underfitting.
