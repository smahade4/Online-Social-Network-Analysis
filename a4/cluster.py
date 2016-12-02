

import matplotlib.pyplot as plt
import os
import pickle
import networkx as nx
from collections import Counter

def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')
    

def example_graph():
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def partition_girvan_newman(graph):
    v={}
    graphcopy=graph.copy()
    v=nx.edge_betweenness_centrality(graph)
        
    while nx.number_connected_components(graphcopy)<3:           
        maxdepth={}
        maxval=0.0 
        for n in v:         
         if graphcopy.has_edge(n[0],n[1]) or graphcopy.has_edge(n[1],n[0]):
            if float(v[n])>=float(maxval):
                if float(v[n])==float(maxval):
                    if n[0]<maxdepth['val'][0]:
                        maxdepth['val']=n
                        maxval=float(v[n])
                         
                    if n[0]==maxdepth['val'][0]:
                        if n[1]<maxdepth['val'][1]:
                            maxdepth['val']=n
                            maxval=float(v[n])
                   
                else:
                   maxdepth['val']=n
                   maxval=float(v[n])
                     
        if graphcopy.has_edge(maxdepth['val'][1],maxdepth['val'][0]):    
                   graphcopy.remove_edge(maxdepth['val'][1],maxdepth['val'][0])
        if graphcopy.has_edge(maxdepth['val'][0],maxdepth['val'][1]):             
                   graphcopy.remove_edge(maxdepth['val'][0],maxdepth['val'][1])
       
           
    components=[]
    for s in nx.connected_component_subgraphs(graphcopy):
          components.append(s)
    return components,graphcopy
    pass




def draw_network(graph,finaldata, filename):
   
    plt.figure(figsize=(12,12))
      
    lab={}
    for f in finaldata:
      lab[f['screen_name']]=f['screen_name']
    '''
    for f in finaldata:
      if 'friends' in f:  
       for d in f['friends']:  
        lab[d]="common"+d
        
    pos=nx.spring_layout(graph,iterations=50,dim=2,k=0.15,scale=1.0)
    '''
    
    pos=nx.spring_layout(graph)
    nodes = graph.nodes()
    colors=[]
    for u in nodes:
        if(u =='Arsenal'):
            colors.append('b')
        if(u =='ChelseaFC' ):
            colors.append('r')
        if(u =='ManUtd'):
            colors.append('w')
       
    nx.draw_networkx(graph,pos,with_labels=True,labels=lab,alpha=.5, width=.6,
                     node_size=100,node_color=colors)
    
    plt.savefig(filename)
    plt.show()
    ###TODO
    pass

'''
def create_graph(finaldata,friend_counts):
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
    
   
    for user in users:
           for friend in user['friends']:
                if(friend_counts[friend]>1):
                    graph.add_edge(user['screen_name'],friend)
    
    for data in finaldata:
       node1= data[0]
       node2= data[1]
       divval=data[3]/(data[4]+data[3])
       if divval >0.01:
           for val in data[2]:
               if(friend_counts[val]>2):
                   graph.add_edge(node1,val)
                   graph.add_edge(node2,val)
              
    return graph
    ###TODO
    pass

'''

def create_graph(userdata):
    graph = nx.Graph()
    
    for f in userdata:
           for friend in f['connection']:
                    graph.add_edge(f['screen_name'],friend,color='g')
              
    return graph
    ###TODO
    pass


def count_friends(users):
    c = Counter()

    for u in users:
      c.update(u['connection'])
   
    return c
    ###TODO
    pass

        
def main():
    if not os.path.isfile("clusterinput.pkl"):
        print("data not loaded properly")
    else:    
     if os.path.getsize("clusterinput.pkl")==0:
        print("data size not proper")
     else:    
      clusterinput = open("clusterinput.pkl","rb")
      clusteroutput = open("clusteroutput.pkl","wb")
      
      data=pickle.load(clusterinput)
      graph= create_graph(data)

      draw_network(graph,data,'BeforeCluster.jpg')       
      clusters,graphcopy= partition_girvan_newman(graph)   
    
      print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
      print('cluster 2 nodes:')
      print(clusters[1].nodes())
      print('cluster 2 nodes:')
      print(clusters[0].nodes())
      print('cluster 2 nodes:')
      print(clusters[2].nodes())
      
      draw_network(graphcopy,data, "AfterCluster.jpg")
      pickle.dump(clusters,clusteroutput)
if __name__ == '__main__':
    main()
