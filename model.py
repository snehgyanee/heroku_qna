import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

wordnet = WordNetLemmatizer()

df = pd.read_csv('data.csv')
df = df.drop('No.',axis=1)
df = df.rename(columns = {'Questions/ Issue ':'Questions','Answer /Steps to Resolve':'Answers'})


def original_data(df):
    cleaned_question = []
    for i in range(len(df['Questions'])):
        review = re.sub('[^a-zA-Z0-9]', ' ', df['Questions'][i])
        review = review.lower()
        review = review.split()
        review = [wordnet.lemmatize(word) for word in review]
        review = ' '.join(review)
        # print(review)
        cleaned_question.append(review)

    df['cleaned_questions'] = cleaned_question

    cleaned_answers = []
    for i in range(len(df['Answers'])):
        review = re.sub('[^a-zA-Z0-9]', ' ', df['Answers'][i])
        review = review.lower()
        review = review.split()
        review = [wordnet.lemmatize(word) for word in review]
        review = ' '.join(review)
        # print(review)
        cleaned_answers.append(review)

    df['cleaned_answers'] = cleaned_answers
    return df.head()

original_data(df)

cleaned_data_list = list(df['cleaned_questions'])
search_terms = ""
def input_question(search_terms):
    print('Enter you query:')
    search_terms = input()
    review = re.sub('[^a-zA-Z0-9]', ' ', search_terms)
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review]
    search_terms = ' '.join(review)

    doc_vectors = TfidfVectorizer()
    doc_vectors = doc_vectors.fit_transform([search_terms] + cleaned_data_list)
    cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors[1:]).flatten()
    df['cosine_score'] = cosine_similarities

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(cosine_similarities):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    most_similar_question = df['Questions'][highest_score_index]
    most_similar_answer = df['Answers'][highest_score_index]
    print("Your Query: ", most_similar_question, "\n\n", "Here is your solution: ", most_similar_answer)

input_question(search_terms)