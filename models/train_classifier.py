import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet') # download for lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    To load data from DB
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    db=pd.read_sql_table('DisasterResponse', engine)
    X = db.message
    Y = db.iloc[:, 4:]
    col=Y.columns
    
    return X, Y, col
    


def tokenize(text):
    '''
    Function: it will back the root after spliting text
    '''
    
    sentance=re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
#   tokenize
    word= word_tokenize(sentance)
    
    lem=WordNetLemmatizer()
    unwanted=stopwords.words("english")
    
# 
    word=[w for w in word if w not in unwanted]
    
    let = [lem.lemmatize(w) for w in word]
    
    return let


def build_model():
    '''
    To build the model
    '''
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),])
    
    #Pipeline parameter
    
    improve =  {
    'vect__max_df': (0.25, 0.50),
    'clf__estimator__min_samples_split': [5]
#     'vect__ngram_range': (1, 1)
#     'vect__max_features': (None, 5000,10000),
#     'tfidf__use_idf': (True, False)
    }
    gs = GridSearchCV(pipeline, param_grid=improve)
    
    return gs


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    To Evaluste the model
    '''
    prediction_gs= model.predict(X_test)
    
    report_gs=classification_report(Y_test, prediction_gs, target_names=category_names)
    print(report_gs)
    Acc = (prediction_gs == Y_test)
    print("The Accracy: \n",Acc.mean())

def save_model(model, model_filepath):
    '''
    Exporting the model
    '''
    
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
