"""
Created on Thu Dec  1 06:33:09 2016

@author: sushma
"""
import pickle
from collections import Counter

def main():
    
 finalfile=open("summary.txt","w")
 clusterinput = open("clusterinput.pkl","rb")
 users=pickle.load(clusterinput)
 classifyinput = open("classifyinput.pkl","rb")
 messagedata=pickle.load(classifyinput)
 counterdata=Counter()
 counterdata.update(users['screen_name'])
 for val in users:       
     counterdata.update(val['connection'])
    
 finalfile.write("Number of users collected"+len(users),finaldata)
 finalfile.write("Number of users collected"+len(counterdata),finaldata)
 finalfile.write("Number of messages collected"+len(messagedata),finaldata)
 clusteroutput = open("clusteroutput.pkl","rb")
 clusters=pickle.load(clusteroutput)
 total=0
 for i in range(0,len(clusters)):
          total=total+len(clusters[i])
 
 finalfile.write("Number of communities discovered"+len(clusters))
 finalfile.write("Average number of users per community"+total/len(clusters))
 classifyoutput = open("classifyoutput.pkl","rb")
 classify=pickle.load(classifyoutput)
 classifycounter=Counter()
 classifycounter.update(classify)
 finalfile.write("Number of instances for class 0 -Male found"+classifycounter[0])
 finalfile.write("Number of instances for class 1 -Female found"+classifycounter[1])

 
if __name__ == '__main__':
    main()