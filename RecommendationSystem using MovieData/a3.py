# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    tokens=[]
    for i in movies['genres'].tolist():
        tokens.append(tokenize_string(i))
    data = np.insert(movies.values,movies.shape[1],tokens,axis=1)
    header = movies.columns.values.tolist()
    header.append('tokens')
    movies = pd.DataFrame(data,columns=header)
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    unsortedvocab={}
    tf={}
    maxk={}
    df={}
    tfidf={}
    listval=movies['tokens'].tolist()
    doccount=0
    for i in listval:
        maxk[doccount]=0
        for val in i:     
            unsortedvocab.setdefault(val, len(unsortedvocab))
            if (val,doccount) not in tf and val in df:
               df[val]=df[val]+1 
            if (val,doccount) not in tf and val not in df:
               df[val]=1     
            if  (val,doccount) in tf:
                tf[(val,doccount)]=tf[(val,doccount)]+1
            else:
                tf[(val,doccount)]=1
            if maxk[doccount]<tf[(val,doccount)]:
                maxk[doccount]=tf[(val,doccount)]    
        doccount=doccount+1        
    totaldocs=doccount    
    vocab={}        
    for i in sorted(unsortedvocab):
               vocab.setdefault(i,len(vocab))
    doccount=0
    rowlist=[]
     
    for i in listval:     
        indices = []
        data = [] 
        rowval=[]
        for val in i:
             if val in vocab:  
              index = vocab[val]
              indices.append(index)
              tfidf[(val,doccount)]=tf[(val, doccount)] / maxk[doccount] * math.log10(totaldocs/df[val])
              data.append(tfidf[(val,doccount)])
              rowval.append(0)
        matrix=csr_matrix((data,(rowval,indices)),shape=(1,len(vocab)))
        rowlist.append(matrix)
        doccount=doccount+1            
    movies['features']=rowlist
    return (movies,vocab)  
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> LA.norm(a)
    7.745966692414834
    """
    ###TODO
    
    datas=np.dot(a,b.transpose())
    calc=np.linalg.norm(a.toarray())*np.linalg.norm(b.toarray())
    return (datas/calc).data

    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    listval=[]
    for index,row in ratings_test.iterrows():      
       total=0
       countc=0
       flag=0
       ratinglist=[]
       datatest=movies.loc[movies['movieId']==int(row['movieId'])] 
       df=ratings_train.loc[ratings_train['userId']==int(row['userId'])]
       for index,rowdf in df.iterrows(): 
           if row['movieId']!=rowdf['movieId']:
               datatrain=movies.loc[movies['movieId']==int(rowdf['movieId'])]
               cval=cosine_sim(datatrain['features'].iloc[0],datatest['features'].iloc[0]) 
               ratinglist.append(rowdf['rating'])
               
               if cval>0:
                   total=total+cval
                   countc=countc+(cval*rowdf['rating'])
                   flag=1
       if flag==1:           
            finalval=countc/total
            listval.append(finalval)
       else:
            finalval=np.mean(ratinglist)
            listval.append(finalval)

    return  np.asarray(listval)
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])
   
if __name__ == '__main__':
    main()
