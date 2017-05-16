import read_data
import math
from read_data import *

def mergeSort(alist,sort_col):
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf,sort_col)
        mergeSort(righthalf,sort_col)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i][sort_col] > righthalf[j][sort_col]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    return alist

def epsilon(values,col):
    top2=list()

    min=3
    c=0
    for i in range(rw):
        top2.append(values[c+min])
        c+=rw

    one=top2[0][col]
    for i in top2:
        if one-i[col] < 0:
            #print "ep1",one,one-i[col]
            break
        one=i[col]

    return one

