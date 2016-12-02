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
 for val in users:        
     counterdata.update(val['screen_name'])
     counterdata.update(val['connection'])
    
 finalfile.write("Number of users collected "+str(len(users)))
 finalfile.write("\n")
 finalfile.write("Number of users collected "+str(len(counterdata)))
 finalfile.write("\n")
 finalfile.write("Number of messages collected "+str(len(messagedata)))
 finalfile.write("\n")
 clusteroutput = open("clusteroutput.pkl","rb")
 clusters=pickle.load(clusteroutput)
 total=0
 for i in range(0,len(clusters)):
          total=total+len(clusters[i])
 
 finalfile.write("Number of communities discovered "+str(len(clusters)))
 finalfile.write("\n")
 finalfile.write("Average number of users per community "+str(total/len(clusters)))
 finalfile.write("\n")
 classifyoutput = open("classifyoutput.pkl","rb")
 classify=pickle.load(classifyoutput)
 classifycounter=Counter()
 classifycounter.update(classify)
 finalfile.write("Number of instances for class 0 -Male found "+str(classifycounter[0]))
 finalfile.write("\n")
 finalfile.write("Number of instances for class 1 -Female found "+str(classifycounter[1]))
 finalfile.write("\n")
 classifyinstance0 = open("classifyoutputinstance0.pkl","rb")
 classify=pickle.load(classifyinstance0)
 finalfile.write("Example of class 0"+str( classify[0]))
 
 classifyinstance0 = open("classifyoutputinstance0.pkl","rb")
 classify=pickle.load(classifyinstance0)
 classifyinstance1 = open("classifyoutputinstance1.pkl","rb")
 classify=pickle.load(classifyinstance1)
 finalfile.write("\n") 
 finalfile.write("Example of class 1"+str( classify[0]))
 
if __name__ == '__main__':
    main()