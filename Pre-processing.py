
#Nama   : Elsa Aritonang
#NIM    : 215150607111005
#Kelas  : PTI PD-B
#TPO2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#deteksi data kosong dan mengisi data yang kosong

datastudent = pd.read_csv('psd/DM/student_feedback.csv')
dataexp = datastudent['Explains concepts in an understandable way']
print(datastudent)
print(datastudent.describe())
print(dataexp.isna().sum())
print (dataexp.mean())
dataexpisi = dataexp.fillna(value = dataexp.mean())
datastudent['Explains concepts in an understandable way'] = dataexpisi
print(dataexpisi)

datawell = datastudent['Well versed with the subject']
datawellisi = datawell.fillna(value = datawell.mean())
datastudent['Well versed with the subject'] = datawellisi

print('data kosong =', dataexpisi.isna().sum())
print('data kosong terisi =', dataexp.isna().sum()-dataexpisi.isna().sum())

#deteksi outlier dan dapatkan data tanpa outlier

print(dataexp)
dataexpisi = dataexp.fillna(value=dataexp.mean())
print(dataexpisi)
print(dataexpisi.describe())

q1, q3 = np.percentile(dataexpisi, [25, 75])
print(q1)
print(q3)
iqr = q3-q1
bb = q1-(1.5*iqr)
ba = q3+(1.5*iqr)
print('batas bawah : ', bb)
print('batas atas : ', ba)
outlier = datastudent[(dataexpisi<bb)|(dataexpisi>ba)]
print('data-data outlier =', outlier)
databersih = datastudent[(dataexpisi>=bb)&(dataexpisi<=ba)]
print('data bersih dari outlier = ', databersih)
print(dataexpisi.describe())

#normalisasi data dengan minmax

datanorm  = datastudent[['Well versed with the subject', 'Explains concepts in an understandable way']]
print(datastudent)
print(datanorm)
scaler = MinMaxScaler()
scaler.fit(datanorm)
datatrans1 = scaler.transform(datanorm)
print(datatrans1)
scalerstd = StandardScaler()
scalerstd.fit(datanorm)
datatrans2 = scalerstd.transform(datanorm)
print(datatrans2)
