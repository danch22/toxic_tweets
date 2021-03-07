import os
from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import pickle

app = Flask(__name__)

URLPATTERN = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"

def load_model(filename):
    file_mod = open(filename, 'rb')
    return pickle.load(file_mod)


clf = load_model('toxic_comments_lr.pkl')
tf = load_model('tfidf_transform.pkl')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


@app.route('/')
def home():
    return render_template('home1.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        new_tweet = request.form['tweet']
        new_tweet = new_tweet.lower()
        new_tweet = re.sub('@#[A-Za-z]+', ' ', new_tweet)
        new_tweet = re.sub(URLPATTERN, 'URL', new_tweet)

        new_tweet = new_tweet.split()

        new_tweet = [word for word in new_tweet if not word in set(all_stopwords)]
        new_tweet = ' '.join(new_tweet)
        new_corpus = [new_tweet]
        new_X_test = tf.transform(new_corpus).toarray()
        pred = clf.predict(new_X_test)
        return render_template('home1.html', prediction=pred)


if __name__ == '__main__':

    # Heroku provides environment variable 'PORT' that should be listened on by Flask
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
