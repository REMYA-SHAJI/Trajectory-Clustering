import xlrd
import math
Unmarked = 99999
filename = "/home/remya/dataset of rfid/dataset4.xlsx"
workbook =xlrd.open_workbook(filename)
csheet = workbook.sheet_by_index(0)
rw=csheet.nrows
Minpts = math.log(rw)
D = {}

aa1=[]
class Data:
    acode = 0.0
    lat = 0.0
    lon = 0.0
    y1 = 0.0
    y2 = 0.0
    value = 0.0
    clabel = Unmarked
    def __init__(self, acode, lat, lon, y1, y2, value):
        self.lat = lat
        self.lon = lon
        self.y1 = y1
        self.y2 = y2
        self.acode = acode
        self.value = value

for r in range(rw):
    r0 = float(csheet.cell(r,0).value)
    r1 = float(csheet.cell(r,1).value)
    r2 = float(csheet.cell(r,2).value)
    r3 = float(csheet.cell(r,3).value)
    r4 = float(csheet.cell(r,4).value)
    r5 = float(csheet.cell(r,5).value)
    aa1.append([r0,r1,r2,r3,r4])
    D[r] = Data(r0, r1, r2, r3, r4, r5)

print aa1
