# Sentiment Analysis

Sentiment Analysis involves predicting whether a given sentence is positive or negative. This repository was created to understand the performance of different models for sentiment analysis and to gain extensive knowledge in this area.

### Models Used:
- LSTM 
- Encoder Network (Transformers)

## LSTM
LSTM networks perform well on sequence data, capturing information from sentences word by word. Given their proven success in NLP tasks, LSTM was my first choice.

### Dataset
- **IMDB 50k:** This dataset consists of 50,000 movie reviews labeled as positive or negative.

- **Data Preprocessing:** I removed HTML tags, stop words, punctuation, and created lemmatized vectors for each token using GloVe vectors (via SpaCy).

- **Block Size:** I selected a block size based on the 97th percentile of the sentence length distribution in the training data, which is 450. Sentences shorter than this were padded with zeros, while longer sentences were truncated to 450 words.

### Training
- **Model Architecture:**
    - LSTM: (B, T, C) -> (B, T, C) -> (B, T*C) (reshape)
    - LayerNorm, Dropout
    - Linear Layer: (B, T*C) -> (B, 500)
    - Dropout, ReLU
    - Linear Layer: (B, 500) -> (B, 2)

- **Hyperparameters:** 
    - **Learning Rate:** 0.001
    - **Block Size:** 450
    - **Optimizer:** Adam

- **Saving Strategy:** The model is saved if test accuracy improves after each epoch.

### Observations and Results
- **Results:** 
    - **Train Accuracy:** 0.98
    - **Test Accuracy:**  0.82

- **Live Link:** [Live Link](https://sentiment-lstm.streamlit.app/)

- **Kaggle Notebook:** [Kaggle Notebook Link](https://www.kaggle.com/code/afsalali/sentiment-analysis/notebook)

- **Observations and Improvements:** 
    - **Overfitting:** The model achieves good results in training and testing but begins to overfit after a few epochs.
    - **Positional Information:** The model needs positional information for better context understanding.
    - **Context Detection:** Results were decent for test data but struggled with context-based sentiment detection.
    - **Stop Words:** Stop words should not be removed to retain contextual meaning.
    - **Data Volume:** More data is needed for a more generalized model.


Not fully satisfied with these results, I developed another model using the Encoder Network, learning from my earlier mistakes.

## Encoder (Transformers)
The Transformer network, introduced in [**Attention Is All You Need**](https://arxiv.org/pdf/1706.03762), is a major advancement in the ML community. It excels at capturing sequential information and is generally faster with GPU acceleration compared to LSTM.

### Dataset
- **Amazon Reviews:** This dataset contains around 4 million sentiment-labeled reviews of Amazon products.

- **Data Preprocessing:** To help the model understand the importance of each word and even characters, I used BERT tokenization after removing the titles of the reviews and sentences with fewer than 5 words.

- **Block Size:** The block size was set at the 97th percentile of the sentence length distribution in the training data, which is 196. Shorter sentences were padded with zeros, while longer ones were truncated to 196 words.

### Training
- **Model Architecture:**
    - Word Embeddings + Positional Embeddings (B, T, C)
    - Dropout, LayerNorm
    - Encoder Block (B, T, C) repeated N times
    - LayerNorm
    - Linear Layer: (B, T*C) -> (B, C)
    - LayerNorm, GELU, Dropout
    - Linear Layer: (B, C) -> (B, 2)

- **Hyperparameters:** 
    - **Learning Rate:** 3e-6
    - **Block Size:** 196
    - **Optimizer:** AdamW
    - **Batch Size:** 2048
    - **Loss Function:** Cross Entropy

- **Training:**
    - **Network Visualization:** I visualized the initial network to assist in debugging the model initialization.
    - **Data Loader:** Implemented a generator to load data chunks one at a time, optimizing space complexity.
    - **Gradient Accumulation:** Used a gradient accumulation strategy to achieve the desired batch size.
    - **Gradient Norm Clipping:** Applied gradient norm clipping at 1.0 for smoother learning.
    - **Training:** Trained for 2 epochs at a learning rate of 3e-5 for 10 hours, then saved the model and trained again for 2 epochs at 3e-6, loading the model on GPU due to Kaggle session breaks.
    - **Saving Strategy:** The model is saved if test accuracy improves after each epoch.

### Observations and Results
- **Results:**
    - **Train Accuracy:** 0.88
    - **Test Accuracy:** 0.87
    
    (The model used in live deployment and on GitHub is a lightweight 34 MB. See Kaggle notebook [**Version 9**](https://www.kaggle.com/code/afsalali/senti-transformer?scriptVersionId=201140111)).

- **Live Link:** [Live Link](https://sentiment-encoder.streamlit.app/)

- **Kaggle Notebook:** [Kaggle Notebook Link](https://www.kaggle.com/code/afsalali/senti-transformer/notebook?scriptVersionId=202291218)

- **Observations and Improvements:** 
    - **Generalization:** The network generalizes well and performs effectively in sentiment analysis compared to the previous LSTM model.
    - **Context Awareness:** The model takes context into account when predicting sentiment.
    - **Model Efficiency:** The lightweight model (34 MB) also performs well in sentiment predictions.
    - **Performance Enhancement:** Using a scheduled learning rate and a higher batch size could further boost performance.
    - **Diverse Training Data:** Adding data from different domains, like product reviews or social media, can improve adaptability.
    - **More Training:** Extending training for additional epochs may enhance accuracy and generalization.


## Links 
- ### LSTM
    - #### [Live Link](https://sentiment-lstm.streamlit.app/)
    - #### [Notebook Link](https://www.kaggle.com/code/afsalali/sentiment-analysis/notebook)

- ### Encoder
    - #### [Live Link](https://sentiment-encoder.streamlit.app/)
    - #### [LW Notebook Link](https://www.kaggle.com/code/afsalali/senti-transformer?scriptVersionId=201140111)
    - #### [Notebook Link](https://www.kaggle.com/code/afsalali/senti-transformer/notebook?scriptVersionId=202291218)