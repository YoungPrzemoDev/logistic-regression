import csv
import math
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# Funkcja do wczytywania danych z pliku CSV
def read_csv_file(input_file: str) -> List[List[str]]:
    # Otwiera plik, dzieli tekst na linie i zwraca listę list zawierających dane z każdego wiersza
    return [row for row in csv.reader(Path(input_file).read_text().splitlines(), delimiter=';')]

# Ścieżka do pliku CSV
path = 'C:\\Users\\michu\\Desktop\\WAT\\SEMESTR-7\\MED\\Lab2\\Płatki-sniadaniowe-cereals.csv'
# Wczytanie danych z pliku CSV
data = read_csv_file(path)


potas = [280,135,320,330,1,70,30,100,125,190,35,105,45,105,55,25,35,20,65,160,130,120,80,30,25,100,200,190,25,40,45,85,90,100,45,90,35,60,95,40,95,55,95,170,170,160,90,40,130,90,120,260,45,15,50,110,110,240,140,110,30,35,95,140,120,40,55,90,35,230,110,60,25,115,110,60]

quantiles = np.percentile(potas, [25, 50, 75])
print(f"25th percentile: {quantiles[0]}")
print(f"50th percentile (median): {quantiles[1]}")
print(f"75th percentile: {quantiles[2]}")

potassium_ranges = [(0, quantiles[0]),
(quantiles[0] + 1, quantiles[1]),
(quantiles[1] + 1, quantiles[2]),
(quantiles[2] + 1, max(potas))]

print(f"Potassium ranges based on quantiles: {potassium_ranges}")



# Inicjalizacja list i zmiennych do przechowywania danych statystycznych
x1 = 0
x2 = 0
x3 = 0
x4 = 0

m1 = 0
m2 = 0
m3 = 0
m4 = 0


# Przetwarzanie danych i kategoryzacja płatków
for i in range(1, len(data)):
    # Sprawdzenie i zliczenie płatków w poszczególnych kategoriach cukru, które są na środkowej półce
    if int(data[i][9]) <= 43 and data[i][-1] == '1':
        x1 += 1
    elif int(data[i][9]) <= 90 and data[i][-1] == '1':
        x2 += 1
    elif int(data[i][9]) <= 121 and data[i][-1] == '1':
        x3 += 1
    elif int(data[i][9]) <= 330 and data[i][-1] == '1':
        x4 += 1


# Kategoryzacja płatków według zawartości potasu
for i in range(1, len(data)):
    # Zliczanie liczby płatków w każdej kategorii potasu
    if int(data[i][9]) <= 43:
        m1 += 1
    elif int(data[i][9]) <= 90:
        m2 += 1
    elif int(data[i][9]) <= 121:
        m3 += 1
    elif int(data[i][9]) <= 330:
        m4 += 1


# Obliczanie prawdopodobieństw znalezienia płatków na środkowej półce dla każdej kategor

n1 = x1
n2 = x2
n3 = x3
n4 = x4

print(n1)
print(n2)
print(n3)
print(n4)


#Obliczam pt
p1 = n1/m1
p2 = n2/m2
p3 = n3/m3
p4 = n4/m4


# pt/(1-pt)
tmp_1 = p1/(1-p1)
tmp_2 = p2/(1-p2)
tmp_3 = p3/(1-p3)
tmp_4 = p4/(1-p4)


#Obliczam Logit
L1 = math.log(tmp_1)
L2 = math.log(tmp_2)
L3 = math.log(tmp_3)
L4 = math.log(tmp_4)


#Przygotowanie danych do regresji
X = [43,90,121,330]
X = np.array(X)
X = np.vstack((np.ones_like(X), X)).T
L = [L1, L2, L3, L4]
L = np.array(L)


#Obliczenie współczynników regresji logistycznej
#Metoda najmniejszych kwadratów
left = np.linalg.inv(np.dot(X.T, X))
right = np.dot(X.T, L)
b = np.dot(left, right)
#Prognozowanie na podstawie modelu
predict = b[0] + b[1]*330
prob = 1/(1 + math.exp(-predict))

print(prob)
print(abs(b[0]/b[1]))

#Przygotowanie danych do wykresu
y = [i for i in range(0, 630)]
x = []
for i in y:
    predict = b[0] + b[1] * i
    x.append(1/(1 + math.exp(-predict)))

plt.plot(y, x)
plt.xlabel("Zawartość potasu")
plt.ylabel("Prawdopodobieństwo")
plt.show()