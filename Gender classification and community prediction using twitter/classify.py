
from collections import Counter, defaultdict
from itertools import  combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import requests
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import pickle





def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])
    



def tokenize(doc, keep_internal_punct=False, collapse_urls=False, collapse_mentions=False):
    if collapse_urls:
       doc = re.sub('http\S+', 'THIS_IS_A_URL', doc)
    if collapse_mentions:
       doc = re.sub('@\S+', 'THIS_IS_A_MENTION', doc)  
    if keep_internal_punct==True:
       s=re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)',' ', doc.lower()).split()               
  
    if keep_internal_punct==False:
       s=re.sub('\W+',' ', doc.lower()).split()
    return np.array(s)
    pass


def token_features(tokens, feats):
    
    c=Counter()
    c.update(tokens)
    
    for s in c: 
      feats.update({'token='+s:c[s]})
    ###TODO
        
    
    pass


def token_pair_features(tokens, feats, k=3):
    
    ###TODO        
    arr=[]
    t=[]
    for i in range(0,len(tokens)):
        if i+(k-1)>len(tokens)-1:
         break
        else:
         val=i+(k-1)
         for j in range(i,val+1): 
             t.append(tokens[j])     
         if len(t)==k:
              s=list(combinations(t, 2))
              for val in s:
                arr.append('__'.join(val))
                t=[]
    
    c=Counter()
    c.update(arr)
    
    for pairval in sorted(c): 
      feats.update({'token_pair='+str(pairval):c[pairval]})
    pass


def featurize(tokens, feature_fns):
    
    
    feats = defaultdict(lambda:0)
    for func in feature_fns:
     func(tokens,feats)
    return sorted(feats.items())
    pass


def vectorize(tokens_list, feature_fns, vocab=None):
    
    ###TODO 
    countterm={}
    unsortedvocab={}
    indptr = [0]
    indices = []
    data = []
    feats=[]
    for d in tokens_list:
       featureval=featurize(d,feature_fns) 
       feats.append(featureval)
       for s in featureval: 
           if s[0] in countterm:
            countterm[s[0]]=countterm[s[0]]+1  
           else:
            countterm[s[0]]=1              
           unsortedvocab.setdefault(s[0], len(unsortedvocab)) 
    
    if vocab==None:        
      vocab={}         
      for i in sorted(unsortedvocab):
               vocab.setdefault(i,len(vocab))
      for d in feats:
        for s in d : 
          if s[0] in vocab and (s[0] !='this_is_a_mention' or s[0] !='this_is_a_url'): 
              index = vocab[s[0]]
              indices.append(index)
              data.append(s[1])            
        indptr.append(len(indices))
    else: 
       for d in feats:
         finallist=[]  
         for s in d: 
            if s[0] in vocab:
              index = vocab[s[0]]
              finallist.append(s)  
              indices.append(index)
              data.append(s[1])
         for s in vocab.keys():
             if s not in finallist:
              index = vocab[s]  
              indices.append(index)
              data.append(0)
         indptr.append(len(indices)) 
    matrix=csr_matrix((data, indices, indptr), dtype=np.int64)
    return matrix,vocab  
    
    pass

def accuracy_score(truth, predicted):
  
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    
    ###TODO
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        if 0 not in labels[train_ind] or 1 not in labels[train_ind]:
            continue
        else:
            clf.fit(X[train_ind], labels[train_ind])
            predictions = clf.predict(X[test_ind])
            accuracies.append(accuracy_score(labels[test_ind], predictions))
   
    return np.mean(accuracies)  
    pass


def eval_all_combinations(docs, labels, punct_vals, collapse_urls,mentions,
                          feature_fns):
   
    ###TODO

    combs = []
    newcmlist=[]
    docval=[]
    finallist=[]
    
    for i in range(1, len(feature_fns)+1):
        comblist = [list(x) for x in combinations(feature_fns, i)]
        combs.append(comblist)
    for val in combs:
              for s in val:    
                        newcmlist.append(s) 
                    
    for p in punct_vals:              
        for val in collapse_urls:
            for data in mentions:
             docval=[tokenize(d['text']+' '+d['user']['description'] if d['user']['description'] else d['text'],p,val,mentions) for d in docs]    
             for s in newcmlist:      
                       matrix,vocab=vectorize(docval, s)
                       model = LogisticRegression()
                       acc=cross_validation_accuracy(model, matrix, labels, 5)   
                       finaldict={}
                       finaldict.update({'accuracy':acc})     
                       finaldict.update({'punct':p})
                       finaldict.update({'url':val})
                       finaldict.update({'mentions':data})
                       finaldict.update({'features':tuple(s)})     
                       finallist.append(finaldict)
              
    vallist = sorted(finallist, key=lambda k: -k['accuracy'])
    return vallist
                                          
    pass


def plot_sorted_accuracies(results):
    ###TODO
    listval=[]
    for val in results:
        listval.append(val['accuracy'])
    listval= sorted(listval)
    plt.plot(np.arange(len(listval)), listval)
    plt.xlabel('setting', size=14)
    plt.ylabel('accuracy', size=14)
    plt.savefig("accuracies.png")
    pass





def fit_best_classifier(docs, labels, best_result):
    
    
    docval=[tokenize(d['text']+' '+d['user']['description'] if d['user']['description'] else d['text'],best_result['punct'],best_result['url'],best_result['mentions']) for d in docs]
    
    matrix,vocab=vectorize(docval,list(best_result['features']))
    model = LogisticRegression()
    model.fit(matrix, labels)
    return model,vocab   
    pass


def top_coefs(clf, label, n, vocab):
    ###TODO
    listcf=[]
    if label==1:
        coef = clf.coef_[0]
        top_coef_ind = np.argsort(coef)[::-1][:n]
    else:
        coef = clf.coef_[0]
        top_coef_ind = np.argsort(coef)[::1][:n]
    for i in top_coef_ind:
        for s in vocab.items():
            if s[1]==i:            
                top_coef_terms = s[0]
        top_coef = coef[i]
        listcf.append((top_coef_terms ,abs(top_coef)))
    return listcf
        
    pass



   

def get_census_names():
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])    
    return male_names, female_names
    
def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()
            
        
def get_first_name_label(tweet):
    if 'name' in tweet:
        parts = tweet['name'].split()
        if len(parts) > 0:
            return parts[0].lower()
            
            
def sample_tweets(data, male_names, female_names):
    tweets = []
    for r in data:
                 if 'user' in r and 'name' in r['user']:
                    name = get_first_name(r)
                    if name in male_names or name in female_names:
                        tweets.append(r)
                    
    return tweets
    
def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  # If term not present, assign next int.
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  # looking up a key; defaultdict takes care of assigning it a value.
    print('%d unique terms in vocabulary' % len(vocabulary))
    return vocabulary
    
def get_gender(tweet, male_names, female_names):
    name = get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1
 

            
def unknown_tweets(data, male_names, female_names):
    tweets = []

    for r in data:
                 if 'user' in r and 'name' in r['user']:
                    name = get_first_name(r)
                    
                    if name not in male_names and name not in female_names:
                        tweets.append(r)
                    
    return tweets



def parse_test_data(best_result,data, vocab):

    docval=[tokenize(d['text']+' '+d['user']['description'] if d['user']['description'] else d['text'],best_result['punct'],best_result['url'],best_result['mentions']) for d in data]
  
    matrix,vocab=vectorize(docval,best_result['features'],vocab)
   
    return matrix    
    
    pass


        
def main():
    if not os.path.isfile("classifyinput.pkl"):
        print("data not loaded properly")
    else:    
     if os.path.getsize("classifyinput.pkl")==0:
        print("data size not proper")
     else:        
        feature_fns = [token_features, token_pair_features]    
        '''
        maledata = open("male.pkl","rb")      
        femaledata = open("female.pkl","rb")
        '''
        classifyinput = open("classifyinput.pkl","rb")
        data=pickle.load(classifyinput)
        flag=0
        while flag==0: 
            '''     
            male_names=pickle.load(maledata)
            female_names=pickle.load(femaledata)
            '''
            male_names, female_names = get_census_names() 
             
            tweets = sample_tweets(data, male_names, female_names)
            y = np.array([get_gender(t, male_names, female_names) for t in tweets])
            if 0 not in y or 1 not in y:
                continue
            else:
                flag=1
        print(y)   
        results = eval_all_combinations(tweets, y,[True,False],[True,False],[True,False],
                                    feature_fns)
    
        best_result = results[0]
        worst_result = results[-1]
        plot_sorted_accuracies(results)   
        clf, vocab = fit_best_classifier(tweets, y, best_result)    
       
        print('best cross-validation result:\n%s' % str(best_result))
        print('worst cross-validation result:\n%s' % str(worst_result))   
        tweets = unknown_tweets(data, male_names, female_names) 
        print(len(tweets))
        X_test=parse_test_data(best_result,tweets, vocab)
        predictions = clf.predict(X_test)
   
        classifyoutput=open("classifyoutput.pkl","wb")
        print(predictions)
        pickle.dump(predictions,classifyoutput)
        classifyoutputinstance0=open("classifyoutputinstance0.pkl","wb")
        classifyoutputinstance1=open("classifyoutputinstance1.pkl","wb")
        instance0=[] 
        instance1=[]
        '''
        for val in zip(predictions,(t['text']+" "+ t['user']['description']+" "+ t['user']['name'] for t in tweets)):
            instance.add(val)
        '''    
        for val in zip(predictions,(t['user']['name'] for t in tweets)):
           if(val[0]==0):
            instance0.append(val)
           if(val[0]==1):
            instance1.append(val)
            
        pickle.dump(instance0,classifyoutputinstance0)
        pickle.dump(instance1,classifyoutputinstance1)
        print(len(predictions))    
                 
        print('\npositive words:')
        print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5,  vocab)]))
        print('negative words:')
        print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
        
if __name__ == '__main__':
    main()
