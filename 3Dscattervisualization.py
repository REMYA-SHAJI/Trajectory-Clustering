

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import xlrd

a1=[]
a2=[]
a3=[]

p1=[]
filename = "/home/remya/dataset of rfid/modules of stsnn/dataset4snn.xlsx"
workbook =xlrd.open_workbook(filename)
csheet = workbook.sheet_by_index(0)
rw=csheet.nrows

for r in range(rw):
    r0 = float(csheet.cell(r,0).value)
    a1.append(r0)
    r1 = float(csheet.cell(r,1).value)
    a2.append(r1)
    r2 = float(csheet.cell(r,2).value)
    a3.append(r2)
    r3 = float(csheet.cell(r,3).value)
    p1.append([a1,a2])

fig =plt.figure()
ax=fig.add_subplot(111,projection='3d')

print p1
X= a1
Y = a2
Z = a3
ax.scatter(X,Y,Z,c='r',marker='o')
ax.set_xlabel('Lattiude ')
ax.set_ylabel('Longitude')
ax.set_zlabel('Non-spatial ')
plt.show()
