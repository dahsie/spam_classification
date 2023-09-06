
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score,accuracy_score,confusion_matrix

from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
import random
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler

#Fix the random seed in order to make the result reproductible
random.seed=42

#Downlod the stopwords from nltk api
nltk.download('stopwords')

# Load the english stopwords list
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    """ Function for tokenizing and removing stopwords
    Args:
        text : the row text to tokenize and remove stopword
    """
    words = gensim.utils.simple_preprocess(text)
    return [word for word in words if word not in stop_words]

def preprocessing():
    """ Function for loading and preprocessing the dataset
    Return the preprocess dataframe. Each document is turn into a list of tokens 
    """
    messages = pd.read_csv('/home/dah/nlp/spam_classification/dataset/spam_ham_dataset.csv', encoding='utf8')
    messages = messages.drop(labels = ["Unnamed: 0","label"], axis = 1)
    messages.columns = ["text", "label"]
    
    messages['text_clean'] = messages['text'].apply(remove_stopwords)

    return messages

def grid_seach_embadding_algo(param_grid,xtrain,ytrain,xtest,ytest,algo):

    """Transform à test data to numerical vectors. Each word à turn into a vector
        Args:
            param_grid : list of parameters on which a grid seach is made in order to find the best parameters which will be use to embed the text
            xtrain : text data obtain after preprocessing
            ytrain : label of each document
            xtest : text data on  which the grid search parameters are evaluate
            ytest : label of text data
            algo: the algorithm chosen to embef the doc ( here we are "word2vec", "FastText",Doc2vec")
    """
    param_combinations = list(ParameterGrid(param_grid))

    # Pour chaque combinaison de paramètres, entraînez un modèle Word2Vec et évaluez-le
    best_accuracy = 0
    best_params = None
    tagged_data = [TaggedDocument(words=text, tags=[str(i)]) for i, text in enumerate(xtrain)]

    for params in tqdm(param_combinations):
        

        
        if algo=="Doc2Vec":
            d2v_model = Doc2Vec(vector_size=params['vector_size'], window=params['window'], min_count=params['min_count'], epochs=params['epochs'],workers=17)
            d2v_model.build_vocab(tagged_data)
            d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

            # Obtain the vectors of each documents
            xtrain_vect = [d2v_model.infer_vector(text) for text in xtrain]
            xtest_vect = [d2v_model.infer_vector(text) for text in xtest]
        else:
            model=None
            if algo=="FastText":
                model = gensim.models.FastText(xtrain,vector_size=params['vector_size'], window=params['window'], min_count=params['min_count'], epochs=params['epochs'],workers=17)  
            elif algo=="Word2Vec":
                model = gensim.models.Word2Vec(xtrain, vector_size=params['vector_size'], window=params['window'], min_count=params['min_count'], epochs=params['epochs'],workers=17)
            
            #avarage the differents vectors of each doc to obtain a single vector for each document
            xtrain_vect = [np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0) for sentence in xtrain]
            xtest_vect = [np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0) for sentence in xtest]

        #Use the random forest classifier the estimate the best parameters obtain form the embedding algorithm
        classifier = RandomForestClassifier()
        classifier.fit(xtrain_vect, ytrain)
        
        y_pred = classifier.predict(xtest_vect)
        
        accuracy = accuracy_score(ytest, y_pred)
        
        # Update the best parameters if the actual parameter is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print("Meilleurs paramètres:", best_params)
    print("Meilleure précision:", best_accuracy)

    return best_params,best_accuracy


def grid_seach_RandomForestClassifier(param_grid,xtrain_vect,ytrain,xtest_vect,ytest):
    """Function to make a grid seach on order to find the random forest best parameters
        Args:
            param_grid : list of parameters on which a grid seach is made in order to find the best parameters which will be use to fit our classifier
            xtrain_vect : list of vectors which will be use to train the model. Each vector represents an entire doc ou message
            xtest_vect : list of vectors which will be use to evaluate the model. Each vector represents an entire doc ou message
            ytrain: label of traing dataset
            ytest : label of evaluation dataset

        Return the best parameters and the the best model


    """
    rf = RandomForestClassifier()

    # Use GridSearchCV to search the best parameters
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(xtrain_vect, ytrain)

    # Obtain the best parameters and the best score
    best_params = grid_search.best_params_
    best_rf_model = grid_search.best_estimator_

    y_pred = best_rf_model.predict(xtest_vect)

    accuracy = accuracy_score(ytest, y_pred)
    print("Meilleurs paramètres Random Forest:", best_params)
    print("Meilleure précision:", accuracy)

    return best_params,best_rf_model

def visualisation(xtrain_vect,ytrain,method,n_component=2):
    """Function for visualizing the embedding vectors.
    
    Args:
        xtrain_vect (list): List of vectors which will be used to train the model. Each vector represents an entire document or message.
        ytrain: Label of the training dataset.
        method (str): Method for reducing the dimensionality of the training vectors (here we are using only two methods: "pca" and "tsne").
        n_component (int): Number of components to preserve.
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(xtrain_vect)

    model=None
    if method=='tsne':
        model=TSNE(n_components=n_component,random_state=42,verbose=2)

    elif method=='pca':
        model=PCA(n_components=n_component,random_state=42)

    xtrain_new=model.fit_transform(X_norm)

    fig = px.scatter(x=xtrain_new[:, 0], y=xtrain_new[:, 1],color=ytrain)
    fig.update_layout(
        title="visualization of word embadding vector",
        xaxis_title="First component",
        yaxis_title="Second component",
    )
    fig.show()


def score(model,xtest_vect,ytest):
    """Function for scoring the model after training
    """
    y_pred = model.predict(xtest_vect)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    accuracy=accuracy_score(ytest,y_pred)
    print('Precision: {}'.format(round(precision, 3)))
    print('Recall: {}'.format(round(recall, 3)))
    print('Accuracy: {}'.format(round(accuracy,3)))
    print(f"confusion matrix { confusion_matrix(ytest,y_pred)}")


def main():

    """The main functions 
    """
    random.seed=42

    
    messages=preprocessing()
    xtrain, xtest, ytrain, ytest = train_test_split(messages['text_clean'],messages['label'], test_size=0.2,random_state=42)
    

    param_grid = {
        'vector_size': [50, 100, 150],  # Taille du vecteur de mot
        'window': [3, 5],            # Fenêtre de contexte
        'min_count': [1, 2, 3],         # Nombre minimum d'occurrences d'un mot
        'epochs': [10, 20, 30,40]         # Nombre d'itérations sur les données
    }


    # grid_seach_embadding_algo(param_grid,xtrain,ytrain,xtest,ytest,"FastText")

    # paramtre for random forest 
    # Meilleurs paramètres: {'epochs': 30, 'min_count': 3, 'vector_size': 150, 'window': 3}
    # Meilleure précision: 0.9806763285024155

    model = gensim.models.FastText(xtrain,vector_size=150,window=3,min_count=3,epochs=30,workers=17,seed=42)

    param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [None, 2, 3],
    'min_samples_split': [2, 5],
    # 'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'criterion' :['gini', 'entropy', 'log_loss'],
    }

    
    xtrain_vect = [np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0) for sentence in xtrain]
    xtest_vect = [np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0) for sentence in xtest]

    #grid_seach_RandomForestClassifier(param_grid_rf,xtrain_vect,ytrain,xtest_vect,ytest)

    #visualisation(xtrain_vect,ytrain,method='tsne',n_component=2)

    # # Créez un modèle MLP Classifier
    # classifier = RandomForestClassifier()
    # classifier.fit(xtrain_vect, ytrain)

    # # Créez un modèle MLP Classifier
    classifier = MLPClassifier(hidden_layer_sizes=(400,100,5), max_iter=150, random_state=42,activation='relu',verbose=True,solver='adam',learning_rate="adaptive",shuffle=True)
    # Entraînez le modèle sur les données d'entraînement
    classifier.fit(xtrain_vect, ytrain)


    
    score(classifier,xtest_vect,ytest)
 
    


if __name__=="__main__":
    main()