import pickle as pk
import streamlit as st
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load models
lr = pk.load(open('LogisticRegression2.pkl', 'rb'))
sgd = pk.load(open('SGD2.pkl', 'rb'))
nb = pk.load(open('NaiveBayes2.pkl', 'rb'))
pa = pk.load(open('PassiveAggressive2.pkl', 'rb'))
ri = pk.load(open('RidgeRegression2.pkl', 'rb'))
vectorizer = pk.load(open('Scaler.pkl', 'rb'))


def clean_review(review):
    return ' '.join(word for word in review.split() if word.lower() not in stopwords.words('english'))


review = st.text_input('Enter Movie Review')
model_choice = st.selectbox('Choose Model', [
                            'SGD', 'Logistic Regression', 'Naive Bayes', 'Passive Aggressive', 'Ridge Regression'])

if st.button('Predict'):
    if not review.strip():
        st.warning('Please enter a review.')
    else:
        cleaned = clean_review(review)
        review_vec = vectorizer.transform([cleaned])
        if model_choice == 'SGD':
            result = sgd.predict(review_vec)
        elif model_choice == 'Logistic Regression':
            result = lr.predict(review_vec)
        elif model_choice == 'Naive Bayes':
            result = nb.predict(review_vec)
        elif model_choice == 'Passive Aggressive':
            result = pa.predict(review_vec)
        else:
            result = ri.predict(review_vec)
        st.write('Positive Review' if result[0] == 1 else 'Negative Review')
