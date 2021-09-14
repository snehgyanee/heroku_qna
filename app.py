import re

import nltk
import pandas as pd
from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import numpy as np
wordnet = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('punkt')
from nltk.tokenize import word_tokenize

df = pd.read_csv('data2.csv')
df = df.drop('No.',axis=1)
df = df.rename(columns = {'Questions/ Issue ':'Questions','Answer /Steps to Resolve':'Answers'})

cleaned_question = []
for i in range(len(df['Questions'])):
    review = re.sub('[^a-zA-Z0-9]', ' ', df['Questions'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review]
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    # print(review)
    cleaned_question.append(review)

df['cleaned_questions'] = cleaned_question

cleaned_data_list = list(df['cleaned_questions'])

app = Flask(__name__)


@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/simi',methods = ['POST'])
def simi():
    if request.method == 'POST':
        search_terms = request.form['query']
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
        #print("Your Query: ", most_similar_question, "\n\n", "Here is your solution: ", most_similar_answer)
        if highest_score < 0.1:
            prediction_answer = 'No answer found'
            prediction_question = 'Invalid question! Please enter a valid question'
        else:
            prediction_answer = most_similar_answer
            prediction_question = most_similar_question

    return render_template('index.html',answer = [prediction_question,prediction_answer])


if __name__ == '__main__':
	app.run(debug=True)