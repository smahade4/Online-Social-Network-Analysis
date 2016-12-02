from TwitterAPI import TwitterAPI,TwitterOAuth

import pickle
import time
import sys

from collections import Counter
import requests

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

'''
def create_graph_otherdata(userdata):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)
        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.
    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph = nx.Graph()
    
    for f in userdata:
           for friend in f['friends']:
                    graph.add_edge(f['screen_name'],friend)
    
    draw_network(graph,userdata,'datagraph')
         
    return graph
    ###TODO
    pass
'''
def count_friends(users):
    c = Counter()

    for u in users:
      c.update(u['connection'])
   
    return c
    ###TODO
    pass
'''
def draw_network(graph,finaldata, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
   
    plt.figure(figsize=(12,12))
      
    lab={}
    for f in finaldata:
      lab[f['screen_name']]=f['screen_name']
    for f in finaldata:
      for d in f['friends']:  
       lab[d]=d
    pos=nx.spring_layout(graph,iterations=50,dim=2,k=0.15,scale=1.0)
    pos=nx.spring_layout(graph)
    
    nx.draw_networkx(graph,pos,with_labels=True,labels=lab,alpha=.5, width=.1,
                     node_size=100)
    
    plt.savefig(filename)
    plt.show()
    ###TODO
    pass
'''
def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    response=robust_request(twitter,"users/lookup", {'screen_name': screen_names})
    user_data=[]
    for r in response:
     user_data.append(r)
    return user_data
    ###TODO
    pass



def get_followers(twitter,users,countval):
  for val in users: 
     cursor=-1 
     followlist=[]
     request =robust_request(twitter,'followers/list', {'screen_name': val['screen_name'],'cursor':cursor,'count':countval})           
     for s in request: 
             if s['screen_name'] not in followlist:
                 followlist.append(s['screen_name'])
     print(len(followlist))             
     val.update({'connection':followlist})            
  return users
  
def get_friends(twitter,users,countval):
  for val in users: 
    followlist=[]  
    tweetlist=[]
    request =robust_request(twitter,'friends/list', {'screen_name': val['screen_name'],'count':countval})           
    for s in request: 
        if s['screen_name'] and s['screen_name'] not in followlist:
                 followlist.append(s['screen_name'])
    if 'connection' in val:             
        val['connection']= val['connection']+followlist                               
    else:
        val['connection']=followlist
    
    request =robust_request(twitter,'statuses/user_timeline', {'screen_name': val['screen_name'],'count':countval})           
    for s in request: 
        tweetlist.append(s)
    print(len(followlist))

  return users
              
        
def main():
    clusterinput = open("clusterinput.pkl","wb")
    classifyinput=open("classifyinput.pkl","wb")
    censusdata_male= open("male.pkl","wb")

    censusdata_female= open("female.pkl","wb")
    data=open("countdata.pkl","wb")
    
    tweetlist=[]
    o = TwitterOAuth.read_file('credentials.txt')
    
    twitter = TwitterAPI(o.consumer_key,
                 o.consumer_secret,
                 o.access_token_key,
                 o.access_token_secret)
     
    twitter=TwitterAPI('KUtk2n721gpZOdNzdofhNMwQ2', 'PCGvOXPdUl9zXqPmZFELHDhyjcpxbjJkhVHEOSiPWjM36SV8Ue', '803658132748697600-RBmP0GByvVEZwNSuaFetd2E0SoSXeKs', 'YJon3mIji7LWmzXAFcEVv0bbLNeYAhDiUT9fEuAUCqind')
    '''
    twitter=TwitterAPI('LtTeYMjefY2vQCSecvwdpQQuf', 'WMi3J6hLmVbNIMA6hWQpeDdmVT9N9gP2NC5VIRfkDnS8Ihyiu9', '1419710478-Vt2i4JqTgk7DLBTA8wwjOagDdtyNmzrgVcP8W9u', '56dIlebPVMbbfIukwEoo1X1zEhQLgGAvm6VFwVXyK6Dc6')
    '''
    screen_names=['Arsenal','ChelseaFC','ManUtd']
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    users =get_followers(twitter,users,10)
    users=get_friends(twitter,users,10)   
    since_id=0
    namelist=[]
    
    while len(tweetlist)<80:
      for data in screen_names: 
       request =robust_request(twitter,'search/tweets', {'q': '@'+data,'count':100,'since_id':since_id,"lang": "en"})           
       for s in request:
         if(s['retweeted']==False and s['user']['screen_name'] not in namelist and  s['user']['description']):  
           tweetlist.append(s)
           namelist.append(s['user']['screen_name'])
           if(since_id<s['id']):
             since_id=s['id']
    print(len(tweetlist))       
    friend_counts = count_friends(users)
    datalist=[]   
    for user in users:      
     for friend in user['connection']:    
       if(friend_counts[friend]>1 and friend not in datalist):
           datalist.append(friend)
           
    user1 = sorted(get_users(twitter, datalist), key=lambda x: x['screen_name'])      
    user1 =get_friends(twitter,user1,10)
  
    users.extend(user1)
     
    pickle.dump(users, clusterinput)
    
    pickle.dump(tweetlist, classifyinput)
    male_names, female_names = get_census_names() 
     
    pickle.dump(male_names,censusdata_male)
    pickle.dump(female_names,censusdata_female)
      
if __name__ == '__main__':
    main()
