import math
import numpy
import numpy as np
from dunnfast1 import *
from davisindex1 import * 
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

import estimate_ep1_ep2
import math
from estimate_ep1_ep2 import *
Noise = 999999
Unmarked = 99999
D = {}
C = {}
X = {}
T = {}
K = {}
Cluster_Label = 0

count = 0
stack = 0

print(Minpts)


def ret_eps(x,y):
    #print x.y1,x.y2,y.y1,y.y2
    for i in new_list:
        #print i
        if i[0]==x.y1 and i[1]==x.y2 and i[2]==y.y1 and i[3]==y.y2: 
            return i[4],i[5]
    
def Retrieve_Neighbours(x, n):
    T = {}
    Y = {}
    count = 0
    for r in range(rw):
        if n == r:
            continue
        else:
            e2,e1 = ret_eps(x, D[r])
            if(e1 <eps1 and e2 < eps2):
                Y[count] = Data(D[r].acode, D[r].lat, D[r].lon,  D[r].y1, D[r].y2, D[r].value)
                global T
                T[count] = r
                #print "t",T
                count += 1
               # print(count)
    #my_file.write(str(r)+ "\n")
    return Y
    

def push(x):
     Y = {}
     global T
     global stack
     for j in range(count):
         Y[stack] = Data(D[T[j]].acode, D[T[j]].lat, D[T[j]].lon,  D[T[j]].y1, D[T[j]].y2, D[T[j]].value)
         K[stack] = T[j]
         stack = stack + 1

def pop():
    stack = stack - 1
    return Y[stack]

        
for r in range(rw):
    r0 = float(csheet.cell(r,0).value)
    r1 = float(csheet.cell(r,3).value)
    r2 = float(csheet.cell(r,4).value)
    r3 = float(csheet.cell(r,1).value)
    r4 = float(csheet.cell(r,2).value)
    r5 = float(csheet.cell(r,5).value)
    D[r] = Data(r0, r1, r2, r3, r4, r5)
    
for r in range(rw):
    if D[r].clabel == Unmarked:
        X = Retrieve_Neighbours(D[r], r)
        #print "calc",count,len(X)
        count = len(X)
        global T
        if count < Minpts:
            D[r].clabel = Noise
        else :
            Cluster_Label = Cluster_Label + 1
            for j in range(count):
               # print T
                D[T[j]].clabel = Cluster_Label
                #print(D[T[j]].acode, D[T[j]].clabel)
                push(D[T[j]])

                while(stack < 0):
                    CurrentObj = pop()
                    X = Retrieve_Neighbors(CurrentObj, K[stack])

                    if count < Minpts:
                        for j in range(count):
                            if D[T[j]].clabel != Noise or D[T[j]].clabel == Unmarked:

                                D[T[j]].clabel = Cluster_Label
                                Push(D[T[j]])
                
points=[]
points2=[]
points1=[]
er=[]


for i in range(0,rw):
    if D[i].clabel != Unmarked and D[i].clabel != Noise:
        points2.append(D[i].clabel)
        points.append([D[i].lat,D[i].lon])
        er.append([D[i].lat,D[i].lon,D[i].clabel])
    

xaxis=list()
yaxis=list()
cluster=list()



        
print er
dict = {}
'''
for l2 in er:
    dict[l2[2]] = l2[0:2]

for i,j in dict.items():
    print j
'''
"+++++++++++++++++++++++++++++ spatial ++++++++++++++++++++++++++"

for i in range(0,rw):
    if D[i].clabel != Unmarked and D[i].clabel != Noise:
        points2.append(D[i].clabel)
        points.append([D[i].y1,D[i].y2])
        er.append([D[i].y1,D[i].y2,D[i].clabel])
    

xaxis=list()
yaxis=list()
cluster=list()



        
#print er
dict = {}



for i in range(rw):
    #print rw
    if D[i].clabel in dict.keys():
        dict[D[i].clabel].append((D[i].y1,D[i].y2))
    else:
        dict[D[i].clabel] = [(D[i].y1,D[i].y2)]


centroide = list()
pts = list()

#print c
for i,j in dict.items():
    c1 = 0
    c2 = 0
    count = 0
    for x in j:
        c1 = c1 + x[0]
        c2 = c2 + x[1]
        count = count + 1
        #centroide = (sum(j[0])/len(j),sum(j[1])/len(j))
    #print c1/count,c2/count
    centroide.append((c1/count,c2/count))
    pts.append(j)
    print "Centroid = ", centroide



diff=[]
clus1=[]
help=0
answer=[]
k=0;

for i,j in dict.items():
    c1=0
    c2=0
    c3=0
    print len(j)
    print "next....."
    x=centroide[k]
    for y in j:
            c1=y[0]-x[0]
            #print "fcgvbyvalue.....",x[0],x[1]
            #print "c1,,,,,,,,,,,,,,,",c1
        
            c2=y[1]-x[1]
            #print "c2,,,,,,,,,,,,,,,",c2
            c3=c3+(abs(c1)**2+abs(c2)**2)
            clus1.append((c3))
    #print"cluster1",clus1
    #print"-----------1st---------",c3
    diff.append((c3))
    k+=1
    
    
    
print diff


silhouette_avg = silhouette_score(points,points2)
print("The average silhouette_score is :", silhouette_avg)
z=numpy.random.rand(Cluster_Label+1)
print("The average dunn index is :", dunn_fast(points,points2))
print("The average davisbouldin index is :", davisbouldin(pts,centroide))


'''============
for i in range(rw):
    print D[i].y1,D[i].y2,D[i].clabel
    xaxis.append(D[i].y1)
    yaxis.append(D[i].y2)
    ,,,,,,,
    if D[i].clabel == Unmarked:
        cluster.append(0.999999)
    else:
        #cluster.append(z[D[i].clabel])
    .......

kl=len(xaxis)
print kl
//////
colors = np.random.rand(5)
area = np.pi * (15 * np.random.rand(5))**2
#z=[0.50365044, 0.95872681, 0.95872681, 0.95872681, 0.50365044, 0.42095497, 0.42095497, 0.42095497, 0.50365044, 0.53476337, 0.53476337, 0.53476337, 0.50365044, 0.20416515, 0.20416515, 0.20416515]
.....
#print cluster
z=numpy.random.rand(100)
plt.scatter(xaxis,yaxis, c=z)

plt.show()
    
'''
