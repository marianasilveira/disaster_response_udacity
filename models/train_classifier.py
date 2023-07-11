# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Regex expression
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    Load data from SQLite database
    Input:
        - database_filepath - string - path to database
    Returns:
        - X - pandas series - messages
        - Y - dataframe - categories
        - category_names - list - category labels
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)
    X = df['message']
    Y = df.iloc[:,4:-1] 
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    Normalize, tokenize and lemmatize text
    Input:
        - text - string
    Returns:
        - clean_tokens - list - clean tokens
    '''
    # Remove urls (if any)
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]"," ", text)
  
    # Get tokens  
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # Intantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize 
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build machine learning pipeline
    Returns:
        - model - GridSearchCV pipeline - machine learning model
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ]) 

    parameters = { 
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [2, 3]
        }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predict and evaluate machine learning model
    Input:
        - model - GridSearchCV pipeline - machine learning model
        - X_test - pandas series - test data
        - Y_test - dataframe - target
        - category_names - list - category names 
    '''
    Y_pred = model.predict(X_test)

    for i,category in enumerate(category_names):
        print('{}: ---------------------------'.format(category))
        y_target = Y_test[category]
        y_pred = Y_pred[:,i]
        print(classification_report(y_target, y_pred))
    
    #print('Model best parameters:' + model.best_params_)
      

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
            
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()