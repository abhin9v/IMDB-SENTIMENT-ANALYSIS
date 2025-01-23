import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

#mapping of words index back to words
word_index=imdb.get_word_index()
#word_index
reverse_word={value:key for key,value in word_index.items()} 

model=load_model("simple_rnn_imdb.h5")


def decode_review(encoded_review):
    return ' '.join([reverse_word.get(i-3,'?')for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#prediction 

#prediction

def predict_sentiment(review):
    preprocessed= preprocess_text(review)

    prediction=model.predict(preprocessed)
    
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

import streamlit as st
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review")

input=st.text_area("Movie Review")

if st.button("Classify"):
    preprocess_text=preprocess_text(input)

    #prediction
    prediction=model.predict(preprocess_text)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"

    st.write(f"sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Enter a review and click on the classify button")
