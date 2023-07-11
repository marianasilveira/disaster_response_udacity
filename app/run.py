import json
import plotly
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib


import os 

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath = "data/DisasterResponse.db"
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table(database_filepath, con=engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # First graph - Frequency of each Category used as Training Data
    # Extract plot information from dataframe
    category_names = df.iloc[:,4:40].columns
    category_names = [name.replace('_',' ').capitalize() for name in category_names]
    
    category_counts = list()
    for column in df.iloc[:,4:40].columns:
        category_counts.append(df[column][df[column] == 1].count())  
        
    # Sort
    category_names_1g = [category_names[i] for i in np.argsort(category_counts)]
    category_counts.sort()

    # Second graph - classification of messages with 'help'
    df['message'] = df['message'].str.lower()
    assistance_msg = df['message'].str.contains('water')|df['message'].str.contains('flood')|df['message'].str.contains('storm')
    assistance_count = list()

    for column in df.iloc[:,4:40].columns:
        assistance_count.append(df[column][assistance_msg & df[column]==1].sum())

    #Sort 
    category_names_2g = [category_names[i] for i in np.argsort(assistance_count)]
    assistance_count.sort()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # First Graph - Frequency of each Category used as Training Data
        {
            'data': [
                Bar(
                    x=category_names_1g,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Frequency of categories used as training data',
                'yaxis': {
                    'title': "Occurrences"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        # Second Graph - Frequency of each Category used as Training Data
        {
            'data': [
                Bar(
                    x=category_names_2g,
                    y=assistance_count
                )
            ],

            'layout': {
                'title': "Frequency of categories when the words 'water', 'storm' or 'flood' appeared",
                'yaxis': {
                    'title': "Occurrences"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()