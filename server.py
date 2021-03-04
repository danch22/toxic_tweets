import os
from flask import Flask, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# load the model from disk


app = Flask(__name__)


def load_model(filename):
    file_mod = open(filename, 'rb')
    return pickle.load(file_mod)


clf = load_model('toxic_comments_lr.pkl')
tf = load_model('tfidf_transform.pkl')


@app.route('/')
def home():
    # return render_template('home.html')
    print('hello world!')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tf.transform(data).toarray()
        my_prediction = clf.predict(vect)
    # return render_template('home.html', prediction=my_prediction)
    return str(my_prediction)


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
