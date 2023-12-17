from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# membaca file csv
weather = pd.read_csv("Laporan Akhir/seattle-weather (1).csv")

print('Sebelum dilakukan pengisian: \n', weather.isna().sum())

# kolom precipitation
precip = weather['precipitation']
precipisi = precip.fillna(value = precip.mean())
weather['precipitation'] = precipisi

# kolom temp_max
tempx = weather['temp_max']
tempxisi = tempx.fillna(value = tempx.mean())
weather['temp_max'] = tempxisi

# kolom temp_min
tempm = weather['temp_min']
tempmisi = tempm.fillna(value = tempm.mean())
weather['temp_min'] = tempmisi

# kolom wind
wind = weather['wind']
windisi = wind.fillna(value = wind.mean())
weather['wind'] = windisi

print('Setelah dilakukan pengisian: \n', weather.isna().sum())

# memisahkan data variabel bebas (X) dan variabel terikat 
feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind']
X = np.array(weather[feature_names])
Y = np.array(weather['weather'])

# membagi data menjadi data training 0,6 (60%) dan data testing 0,4 (40%) dengan dirandom 5 kali
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 5)

# preproses data dengan normalisasi
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)

# proses klasifikasi KNN
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)

df_datahasil = pd.DataFrame(X_test, columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
df_datahasil['Kelas Aktual']=Y_test
df_datahasil['Kelas Sistem']=np.array(pred)
print(df_datahasil)

# hasil evaluasi KNN
print('Accuracy of K-NN classifier on test set: ', knn.score(X_test, Y_test))
print(confusion_matrix(Y_test, pred))
hasilpengujian = classification_report(Y_test, pred)
print(hasilpengujian)

# prediksi = data di luar data yang saat ini kita punya
observasi = [[0.5, 5.48, 3.54, 0.13]]
y_predobser = knn.predict(observasi)
print(y_predobser)
