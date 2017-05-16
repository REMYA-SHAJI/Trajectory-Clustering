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

a1=[]
a2=[]
a3=[]
for i in range(0,rw):
    if D[i].clabel != Unmarked and D[i].clabel != Noise:
        a1.append(D[i].y1)
        a2.append(D[i].y2)
        a3.append(D[i].lat)
print a1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig =plt.figure()
ax=fig.add_subplot(111,projection='3d')

X= a1
Y = a2
Z = a3
ax.scatter(X,Y,Z,c='r',marker='o')
ax.set_xlabel('Lattiude ')
ax.set_ylabel('Longitude')
ax.set_zlabel('Non-spatial ')
plt.show()


