import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

df = pd.read_csv('/content/revised_reviews.csv')
x = df['review']
y = df['label']
st.title("REVIEW CLASSIFIER")
st.subheader('TFIFD Vectorizer')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

final = Pipeline([('vect',TfidfVectorizer()),('model',SVC())])
final.fit(x_train,y_train)
t = st.text_input("Enter a review")
message = final.predict([t])
if st.button("Predict"):
   st.title(message)
  