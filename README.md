# Sentiment Analysis with LSTM

## Overview

This project focuses on sentiment analysis using a Long Short-Term Memory (LSTM) network. The goal is to classify tweets as either positive or negative. The model preprocesses the text data, tokenizes it, and then feeds it into an LSTM network to learn and predict sentiments.

## Installation

To run this project, ensure you have the following dependencies installed:

```sh
pip install numpy pandas scikit-learn keras tensorflow
```

## Data Preprocessing

1. **Load Data:**

   ```python
   import pandas as pd

   data = pd.read_csv('Sentiment.csv')
   data = data[['text', 'sentiment']]
   ```

2. **Filter Data:**

   ```python
   data = data[data.sentiment != "Neutral"]
   data['text'] = data['text'].apply(lambda x: x.lower())
   data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
   ```

3. **Tokenization and Padding:**

   ```python
   from keras.preprocessing.text import Tokenizer
   from keras.preprocessing.sequence import pad_sequences

   max_features = 2000
   tokenizer = Tokenizer(num_words=max_features, split=' ')
   tokenizer.fit_on_texts(data['text'].values)
   X = tokenizer.texts_to_sequences(data['text'].values)
   X = pad_sequences(X)
   ```

## Model Building

1. **Define Model:**

   ```python
   from keras.models import Sequential
   from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

   embed_dim = 128
   lstm_out = 196

   model = Sequential()
   model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
   model.add(SpatialDropout1D(0.4))
   model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
   model.add(Dense(2, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   print(model.summary())
   ```

2. **Train-Test Split:**

   ```python
   from sklearn.model_selection import train_test_split
   import pandas as pd

   Y = pd.get_dummies(data['sentiment']).values
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
   ```

3. **Training:**
   ```python
   batch_size = 32
   model.fit(X_train, Y_train, epochs=7, batch_size=batch_size, verbose=2)
   ```

## Evaluation

1. **Validation:**

   ```python
   validation_size = 1500
   X_validate = X_test[-validation_size:]
   Y_validate = Y_test[-validation_size:]
   X_test = X_test[:-validation_size]
   Y_test = Y_test[:-validation_size]
   score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
   print("score: %.2f" % score)
   print("acc: %.2f" % acc)
   ```

2. **Accuracy Measurement:**

   ```python
   import numpy as np

   pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
   for x in range(len(X_validate)):
       result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]
       if np.argmax(result) == np.argmax(Y_validate[x]):
           if np.argmax(Y_validate[x]) == 0:
               neg_correct += 1
           else:
               pos_correct += 1
       if np.argmax(Y_validate[x]) == 0:
           neg_cnt += 1
       else:
           pos_cnt += 1

   print("pos_acc", pos_correct/pos_cnt*100, "%")
   print("neg_acc", neg_correct/neg_cnt*100, "%")
   ```

## Prediction Example

```python
twt = ['Meetings: Because none of us is as dumb as all of us.']
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
sentiment = model.predict(twt, batch_size=1, verbose=2)[0]
if np.argmax(sentiment) == 0:
    print("negative")
elif np.argmax(sentiment) == 1:
    print("positive")
```

## Notes

- The model performs poorly due to unbalanced data. Consider acquiring more data or using pre-trained models for better performance.
- This project is a beginner-level demonstration of LSTM for text classification.

## Future Work

- Balance the dataset for better performance.
- Experiment with different architectures and hyperparameters.
- Incorporate pre-trained word embeddings for improved results.

## Acknowledgments

This project was initially created as a basic demonstration and has received significant attention. Thank you to everyone who has provided feedback and suggestions.
