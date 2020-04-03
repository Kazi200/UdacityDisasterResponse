import sys
from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    # loads data from SQLlite database in DataFrame
    # splits DataFrame into dependent and independent variables for modelling
    # extracts category names from DataFrame
    # accepts filepath as inputs
    # returns dependent variable, independent variable, and category names

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages',engine)
    X = df.message.values
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.tolist()
    return X, y, category_names


def tokenize(text):
    # splits a string of text into tokens
    # converts individual tokens into lemmatized form
    # converts to lowercase
    # removes whitespace
    # accepts text string
    # returns cleaned tokens

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # Builds a model using pipelines
    # applies a count vectorizer in parallel with a TF-IDF transformation
    # gridsearch is applied to find the best hyperparameters

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    parameters = {
#         'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
#         'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
#         'features__text_pipeline__vect__max_features': (None, 5000, 10000),
#         'features__text_pipeline__tfidf__use_idf': (True)
#         'clf__estimator__min_samples_split': [2, 4],
#         'clf__estimator__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # prints out recall, precision and F1 score for the model built on test set

    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(classification_report(Y_test[i], Y_pred[i], labels = [0,1] ))
    pass


def save_model(model, model_filepath):
    # saves the model in a pickle file
    
    filename =  model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


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
