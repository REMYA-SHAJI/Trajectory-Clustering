import euclidean_distandcopytofile
import math
from euclidean_distandcopytofile import *

a={}
values = []
def euclid1(x, y):
    return math.sqrt((x.lat - y.lat)**2 + (x.lon - y.lon)**2)
def euclid2(x, y):
    return math.sqrt((x.y1 - y.y1)**2 + (x.y2 - y.y2)**2)

new_list=list()
values=list()
for r in range(rw):
    sort_list=list()
    for y in range(rw):
        dis1=euclid1(D[r],D[y])
        dis2=euclid2(D[r], D[y])
        sort_list.append([D[r].lat,D[r].lon,D[y].lat,D[y].lon,dis1,dis2])
        new_list.append([D[r].lat,D[r].lon,D[y].lat,D[y].lon,dis1,dis2])
    values.extend(mergeSort(sort_list,4))

eps1 = epsilon(values,4)
eps2 = epsilon(values,5)


print "EPS1, EPS2= ",eps1,eps2

import csv
csvfile = "/home/remya/dataset of rfid/modules of stdbscan/out/out1.csv"
   
#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(values)

csvfile = "/home/remya/dataset of rfid/modules of stdbscan/out/unsorted.csv"
   
#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(new_list)
