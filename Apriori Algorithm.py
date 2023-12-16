
# Nama   : Elsa Aritonang
# NIM    : 215150607111005
# Kelas  : PTI PD-B
# Tugas Praktik - Algoritma Apriori

import pandas as pd
import numpy as np
from apyori import apriori

# Load data
df = pd.read_excel('transaction.xlsx')  
# Hapus kolom tidak perlu
data=df.drop(['Tanggal','ID Transaksi'],axis=1)
 
# Membuat list dalam list dari transaksi pembelian barang
records = []
for i in range(data.shape[0]):
    records.append([str(data.values[i,j]).split(',') for j in range(data.shape[1])])

trx = [[] for trx in range(len(records))]
for i in range(len(records)):
    for j in records[i][0]:
        trx[i].append(j)
trx

# Menggunakan fungsi apriori untuk membuat asosiasi
association_rules = apriori(trx, min_support = 0.14, min_confidence = 0.80, min_lift = 1)
# Membuat list hasil dari algoritma apriori
association_results = list(association_rules)

# Menampilkan hasil asosiasi dari item 
for item in association_results:    
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
        
    print("Support: " + str(item[1]))
    
    print("Confidence: " + str(item[2][0][2]))
    print("=======================================")
    
print("Banyaknya strong association rules: ", len(association_results))


 