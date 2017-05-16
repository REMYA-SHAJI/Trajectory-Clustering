import math
from dunnfast1 import *
from davisindex import * 
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
points=[]
points1=[]

class kinf:
    def __init__(self, p, dt, nsn, il):
        self.Point = p
        self.distance_to = dt
        self.NumOfSharedNeigh = nsn
        self.is_linked = il


class inf:
    def __init__(self, p, type, coord_x, coord_y, knearest, density, cluster_id):
        self.Point = int(p)
        self.type = type
        self.coord_x = float(coord_x)
        self.coord_y = float(coord_y)
        self.knearest = []
        self.density = -1
        self.cluster_id = cluster_id


class SSNClusters():
    def __init__(self, points_list, neighbors=7, radius=None):
        self.ssn_array = []
        self.K = neighbors

        if radius is None:
            self.EPS = int(self.K * 3 / 10)
        else:
            self.EPS = radius
        self.MinPts = int(self.K * 7 / 10)
        self.MyColec = []
        self.cluster_dict = {}

        count = 1
        for point in points_list:
            p = inf(count, '', point[0], point[1], None, None, -1)
            self.ssn_array.insert(count, p)
            count += 1

    def insertKN(self, i, val):
        kn = self.ssn_array[i].knearest
        kn.insert(len(kn)-1, val)

    def get_knearest(self):
        for i in range(0, len(self.ssn_array)):
            count = 0
            for j in range(0, len(self.ssn_array)):
                if i != j:
                    count += 1
                    dist = math.sqrt(((self.ssn_array[i].coord_x - self.ssn_array[j].coord_x) * (self.ssn_array[i].coord_x - self.ssn_array[j].coord_x))
                                     + ((self.ssn_array[i].coord_y - self.ssn_array[j].coord_y) * (self.ssn_array[i].coord_y - self.ssn_array[j].coord_y)))

                    if count <= self.K:
                        kn = kinf(self.ssn_array[j].Point, dist, None, 0)
                        self.insertKN(i, kn)

                    else:
                        index = self.get_max(self.ssn_array[i].knearest)
                        if self.ssn_array[i].knearest[index].distance_to > dist:
                            self.ssn_array[i].knearest[index].distance_to = dist
                            self.ssn_array[i].knearest[index].Point = self.ssn_array[j].Point
                            self.ssn_array[i].knearest[index].is_linked = 0

            #print"point i = ", self.ssn_array[i].Point
            #print "---spatial"
            #for j in range(len(self.ssn_array[i].knearest)):
                #spatial.append(self.ssn_array[i].knearest[j])
                #print self.ssn_array[i].knearest[j].Point
        self.order_knearest_array()

    def order_knearest_array(self):
        temp = []
        for i in range(0, len(self.ssn_array)):
            for j in range(0, len(self.ssn_array[i].knearest)-1):
                for l in range((j + 1), len(self.ssn_array[i].knearest)):
                    if self.ssn_array[i].knearest[j].distance_to > self.ssn_array[i].knearest[l].distance_to:
                        temp.insert(0, self.ssn_array[i].knearest[j])
                        self.ssn_array[i].knearest[j] = self.ssn_array[i].knearest[l]
                        self.ssn_array[i].knearest[l] = temp[0]

    def shared_nearest(self):
        for i in range(0, len(self.ssn_array)):
            for j in range(0, len(self.ssn_array[i].knearest)):
                count_share = 0
                for l in range(0, len(self.ssn_array)):
                    if self.ssn_array[i].knearest[j].Point == self.ssn_array[l].Point:
                        for n in range(0, len(self.ssn_array[l].knearest)):
                            if self.ssn_array[l].knearest[n].Point == self.ssn_array[i].Point:
                                self.ssn_array[i].knearest[j].is_linked = 1

                        if self.ssn_array[i].knearest[j].is_linked == 0:
                            self.ssn_array[i].knearest[j].NumOfSharedNeigh = 0
                        else:
                            for n in range(0, len(self.ssn_array[l].knearest)):
                                for m in range(0, len(self.ssn_array[l].knearest)):
                                    if self.ssn_array[l].knearest[m].Point == self.ssn_array[i].knearest[n].Point:
                                        count_share += 1

                            self.ssn_array[i].knearest[j].NumOfSharedNeigh = count_share

                            break

    def calculate_density(self):
        for i in range(0, len(self.ssn_array)):
            for j in range(0, len(self.ssn_array[i].knearest)):
                if self.ssn_array[i].knearest[j].NumOfSharedNeigh >= self.EPS:
                    self.ssn_array[i].density = self.ssn_array[i].density + (1 * self.ssn_array[i].knearest[j].is_linked)
                else:
                    self.ssn_array[i].density = self.ssn_array[i].density + (0 * self.ssn_array[i].knearest[j].is_linked)

    def check_cores(self):
        for i in range(0, len(self.ssn_array)):
            if self.ssn_array[i].density >= self.MinPts:
                self.ssn_array[i].type = 'Core'
                self.MyColec.insert(len(self.MyColec), i)
            else:
                self.ssn_array[i].type = 'Border'
                self.MyColec.insert(len(self.MyColec), i)

    def build_clusters(self):
        cluster_id = 0
        for i in range(0, len(self.ssn_array)):
            if self.ssn_array[i].type != 'Noise' and self.ssn_array[i].cluster_id == -1:
                cluster_id += 1
                self.ssn_array[i].cluster_id = cluster_id
                self.cluster_neighbors(self.ssn_array[i].Point, cluster_id)

        for i in range(0, len(self.ssn_array)):
            if self.ssn_array[i].cluster_id > 0:
                if self.ssn_array[i].cluster_id in self.cluster_dict.keys():
                    self.cluster_dict[self.ssn_array[i].cluster_id].append((self.ssn_array[i].coord_x, self.ssn_array[i].coord_y))
                else:
                    self.cluster_dict[self.ssn_array[i].cluster_id] = [(self.ssn_array[i].coord_x, self.ssn_array[i].coord_y)]
        for i in range(0, len(self.ssn_array)):
            points.append([self.ssn_array[i].coord_x, self.ssn_array[i].coord_y])
            points1.append(self.ssn_array[i].cluster_id)
            #print self.ssn_array[i].cluster_id,self.ssn_array[i].coord_x, self.ssn_array[i].coord_y
        return cluster_id

    def cluster_neighbors(self, Point, cluster_id):
        neighbors = []
        index = None
        new_point = None
        for m in range(0, len(self.ssn_array)):
            if self.ssn_array[m].Point == Point:
                neighbors = self.ssn_array[m].knearest # all k's of the ssn_array(m).point
                index = m
                break
        for j in range(0, len(neighbors)):
            new_point = neighbors[j].Point # 1 of the ssn_array(m).point K's
            for l in range(0, len(self.ssn_array)):
                if self.ssn_array[l].Point == new_point:
                    if self.ssn_array[l].type != 'Noise' and self.ssn_array[l].cluster_id == -1 and neighbors[j].NumOfSharedNeigh >= self.EPS:
                        self.ssn_array[l].cluster_id = cluster_id
                        self.cluster_neighbors(new_point, cluster_id)

    def check_similarity(self, i, j):
        result = 0
        total = 0
        for m in range(0, len(self.ssn_array[i].knearest)):
            if self.ssn_array[i].Point == self.ssn_array[j].knearest[m].Point:
                for n in range(0, len(self.ssn_array[i].knearest)):
                    if self.ssn_array[j].Point == self.ssn_array[i].knearest[n].Point:
                        result = self.ssn_array[i].knearest[n].Point
                        total += 1
                        break
        return total
    
    def noise_points(self):
        similarity1 = None
        similarity2 = None
        for i in range(0, len(self.ssn_array)):
            similarity1 = 0
            if self.ssn_array[i].type == 'Border':
                for j in range(0, len(self.MyColec)):
                    similarity2 = self.check_similarity(i, j)
                    if similarity2 > similarity1:
                        similarity1 = similarity2
                if similarity1 < self.EPS:
                    self.ssn_array[i].type = 'Noise'
                    self.ssn_array[i].cluster_id = 0

    def get_max(self, kn):
        max = float
        for idx in range(0, len(kn)):
            if idx == 0:
                max = kn[idx].distance_to
                max_idx = idx
            elif kn[idx].distance_to > max:
                max = kn[idx].distance_to
                max_idx = idx
        return max_idx

    def get_clusters(self):
        self.get_knearest()
        self.shared_nearest()
        self.calculate_density()
        self.check_cores()
        self.noise_points()
        self.build_clusters()
        return self.cluster_dict

import xlrd
import math
#from sklearn.metrics import silhouette_samples, silhouette_score
Unmarked = 99999
Noise = 999999
filename = "/home/remya/dataset of rfid/modules of stsnn/dataset4snn.xlsx"
workbook =xlrd.open_workbook(filename)
csheet = workbook.sheet_by_index(0)
rw=csheet.nrows
pt=[]
for r in range(rw):
    r0 = float(csheet.cell(r,0).value)
    r1 = float(csheet.cell(r,1).value)
    r2 = float(csheet.cell(r,2).value)
    r3 = float(csheet.cell(r,3).value)
    pt.append([r0,r1,r2,r3])
    #print r0,r1
    
c=SSNClusters(pt,9).get_clusters()
'''
range_n_clusters = [31]
for n_clusters in range_n_clusters:
    print points1
    silhouette_avg = silhouette_score(points,points1)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

#print c
'''

centroid = list()
pts = list()
cen=[]
#print c
for i,j in c.items():
    #print "Cluster",i,"---",j
    #print j
    # "n\n\n",len(j),j
    c1 = 0
    c2 = 0
    count = 0
    for x in j:
        #print x[0]
        c1 = c1 + x[0]
        c2 = c2 + x[1]
        count = count + 1
        #centroide = (sum(j[0])/len(j),sum(j[1])/len(j))
    #print c1/count,c2/count
    #print count
    #print (c1/count)
    
    #print "dsfr"
    centroid.append((c1/count,c2/count))
    cen.append([c1/count,c2/count])
    pts.append(j)
    #print "Centroid = ", cen
    range_n_clusters = [i]


diff=[]
clus1=[]
help=0
answer=[]
cent_len=len(centroid)
k=0;

for i,j in c.items():
    c1=0
    c2=0
    c3=0
    print len(j)
    print "next....."
    x=centroid[k]
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


'''

for i,j in c.items():
    c1=0
    c2=0
    for y in j:
        k1 = k1 + y[0]
        k2 = k2 + y[1]
        count = count + 1
        cen1=c1/count
        cen1=c1/count
    cen1=c1/count
    cen2=c2/count
    c1=y[1]-x[1]
    print "c1,,,,,,,,,,,,,,,",c1
    c2=y[1]-x[1]
    print "c2,,,,,,,,,,,,,,,",c2
    c3=c3+(abs(c1)**2+abs(c2)**2)
    clus1.append((c3))

'''




for n_clusters in range_n_clusters:
    #print points1
    silhouette_avg = silhouette_score(points,points1)
    #print "\n\n\n",points,points1
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average dunn index is :", dunn_fast(points,points1))
    print("For n_clusters =", n_clusters,
          "The average davisbouldin index is :", davisbouldin(pts,centroid))
    #silhouette_avg = silhouette_score(j,   i)
    #print("For n_clusters =", i,"The average silhouette_score is :", silhouette_avg)

#print points
#print metrics.calinski_harabaz_score(points, points1)




