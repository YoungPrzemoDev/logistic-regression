import csv
import math
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def read_csv_file(input_file: str) -> List[List[str]]:
    return [row for row in csv.reader(Path(input_file).read_text().splitlines(), delimiter=';')]


path = 'C:\\Users\\michu\\Desktop\\WAT\\SEMESTR-7\\MED\\Lab2\\Płatki-sniadaniowe-cereals.csv'
data = read_csv_file(path)
cukry_1 = []
cukry_2 = []
cukry_3 = []
x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
m1 = 0
m2 = 0
m3 = 0
m4 = 0
m5 = 0
for i in range(1, len(data)):
    if int(data[i][3]) <= 3 and data[i][-4] == 'T':
        x1 += 1
    elif int(data[i][3]) <= 6 and data[i][-4] == 'T':
        x2 += 1
    elif int(data[i][3]) <= 9 and data[i][-4] == 'T':
        x3 += 1
    elif int(data[i][3]) <= 12 and data[i][-4] == 'T':
        x4 += 1
    elif int(data[i][3]) <= 15 and data[i][-4] == 'T':
        x5 += 1

#Cukry
#0 - 3
#4 - 6
#7 - 9
#10 - 12
#13 - 15


#Liczba platków
for i in range(1, len(data)):
    if int(data[i][3]) <= 3:
        m1 += 1
    elif int(data[i][3]) <= 6:
        m2 += 1
    elif int(data[i][3]) <= 9:
        m3 += 1
    elif int(data[i][3]) <= 12:
        m4 += 1
    elif int(data[i][3]) <= 15:
        m5 += 1


#Liczba platków, które są na środkowej półce
n1 = x1
n2 = x2
n3 = x3
n4 = x4
n5 = x5

#Obliczam pt
p1 = n1/m1
p2 = n2/m2
p3 = n3/m3
p4 = n4/m4
p5 = n5/m5
# pt/(1-pt)
tmp_1 = p1/(1-p1)
tmp_2 = p2/(1-p2)
tmp_3 = p3/(1-p3)
tmp_4 = p4/(1-p4)
tmp_5 = p5/(1-p5)

#Obliczam Lt
L1 = math.log(tmp_1)
L2 = math.log(tmp_2)
L3 = math.log(tmp_3)
L4 = math.log(tmp_4)
L5 = math.log(tmp_5)
X = [3, 6, 9, 12, 15]
X = np.array(X)
X = np.vstack((np.ones_like(X), X)).T
L = [L1, L2, L3, L4, L5]
L = np.array(L)

left = np.linalg.inv(np.dot(X.T, X))
right = np.dot(X.T, L)
b = np.dot(left, right)
predict = b[0] + b[1]*15
prob = 1/(1 + math.exp(-predict))

print(prob)
print(abs(b[0]/b[1]))

y = [i for i in range(0, 35)]
x = []
for i in y:
    predict = b[0] + b[1] * i
    x.append(1/(1 + math.exp(-predict)))

plt.plot(y, x)
plt.xlabel("Zawartość cukru")
plt.ylabel("Prawdopodobieństwo")
plt.show()