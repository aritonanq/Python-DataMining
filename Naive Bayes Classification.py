#Nama   : Elsa Aritonang
#NIM    : 215150607111005
#Kelas  : PTI PD-B
#TPO4

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
 
# membaca file csv
weather = pd.read_csv("seattle-weather (1).csv")

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

# memisahkan data variabel bebas (X) dan variabel terikat 
feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind']
X = np.array(weather[feature_names])
Y = np.array(weather['weather'])

# membagi data menjadi data training dan data testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=5)

# preproses data dengan normalisasi
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# proses klasifikasi Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
nilaiprob = classifier.predict_proba(X_test)
y_pred = classifier.predict(X_test)

df_datahasil = pd.DataFrame(X_test, columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
df_datahasil['Kelas Aktual']=Y_test
df_datahasil['Kelas Sistem']=np.array(y_pred)
print(df_datahasil)

# hasil evaluasi Naive Bayes
print('Accuracy of Naive Kontinu classifier on test set: ', classifier.score(X_test, Y_test))
print(confusion_matrix(Y_test, y_pred))
hasilpengujian = classification_report(Y_test, y_pred)
print(hasilpengujian)

# prediksi = data di luar data yang saat ini kita punya
observasi = [[0.5, 5.48, 3.54, 0.13]]
y_predobser = classifier.predict(observasi)
print(y_predobser)
