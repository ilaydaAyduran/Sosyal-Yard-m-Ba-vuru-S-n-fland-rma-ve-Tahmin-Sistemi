import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# CSV dosyasını okuma
df = pd.read_csv(r'csv dosyanızın yolu girilmeli', encoding='latin1', sep=';')

# Sütun isimlerini düzeltme
df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('?', '').str.replace('ı', 'i').str.replace('ç', 'c').str.replace('ü', 'u')

# 'AlinanMaas' sütununu sayısal değerlere dönüştürme
df['AlinanMaas'] = pd.to_numeric(df['AlinanMaas'], errors='coerce')

# Eksik veya hatalı maaş değerlerini medyan ile doldurma
median_value = df['AlinanMaas'].median()
df['AlinanMaas'] = df['AlinanMaas'].fillna(median_value)

# Özellik ve hedef değişkenleri ayırma
X = df.drop(['BasvuruDurum', 'Id', 'TcKimlikNo'], axis=1)
y = df['BasvuruDurum']

# Tüm sütunlarda eksik değer olup olmadığını kontrol etme
print("Eksik değer sayısı:")
print(X.isnull().sum())

# Eğitim ve test setlerine bölme
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kategorik verileri one-hot encoding ile dönüştürme
kategorik_ozellikler = ['MedeniHal', 'Arac']
sayisal_ozellikler = ['Puan', 'KiraTutari', 'kisi', 'AlinanMaas']

on_isleme = ColumnTransformer(
    transformers=[
        ('sayi', StandardScaler(), sayisal_ozellikler),
        ('kat', OneHotEncoder(handle_unknown='ignore'), kategorik_ozellikler)
    ]
)

# KNN modeli pipeline ile oluşturma
boruhattı = Pipeline(steps=[
    ('on_isleme', on_isleme),
    ('siniflandirici', KNeighborsClassifier())
])

# Hiperparametre araması için parametreler
parametreler = {
    'siniflandirici__n_neighbors': [1, 3, 5, 7, 9],
    'siniflandirici__weights': ['uniform', 'distance'],
    'siniflandirici__metric': ['euclidean', 'manhattan']
}

# GridSearchCV ile en iyi parametreleri bulma
grid_arama = GridSearchCV(boruhattı, parametreler, cv=5, scoring='accuracy', n_jobs=-1)

# Eksik değerleri kontrol edip, eksik değerleri eğitimden önce temizle
if X_egitim.isnull().values.any():
    X_egitim = X_egitim.dropna()
    y_egitim = y_egitim[X_egitim.index]  # Eğitim setindeki eksik değerleri hedef değişkende de senkronize edin

if X_test.isnull().values.any():
    X_test = X_test.dropna()
    y_test = y_test[X_test.index]  # Test setindeki eksik değerleri hedef değişkende de senkronize edin

grid_arama.fit(X_egitim, y_egitim)

# En iyi sonucu elde etme
en_iyi_model = grid_arama.best_estimator_
y_egitim_tahmin = en_iyi_model.predict(X_egitim)
y_test_tahmin = en_iyi_model.predict(X_test)

print("En İyi Parametreler:", grid_arama.best_params_)
print("En İyi Çapraz Doğrulama Başarımı:", grid_arama.best_score_)
print("Eğitim Başarımı:", accuracy_score(y_egitim, y_egitim_tahmin))
print("Test Başarımı:", accuracy_score(y_test, y_test_tahmin))
print("Test Seti Sınıflandırma Raporu:")
print(classification_report(y_test, y_test_tahmin))

# Örnek bir kişinin verilerini içeren sözlük
ornek_veri = {
    'Puan':200,          # Örneğin kişinin puanı
    'KiraTutari': "36000",  # Örneğin kira tutarı
    'kisi':"4" ,           # Örneğin hanede yaşayan kişi sayısı
    'AlinanMaas': "50000",  # Örneğin alınan maaş
    'MedeniHal': 'evli',  # Örneğin medeni hali
    'Arac': 'Evet'       # Örneğin araç sahibi mi
}

# DataFrame'e çevirme
ornek_df = pd.DataFrame([ornek_veri])

# Özelliklerin sırasını eğitim setiyle eşleştirme (gerekliyse)
ornek_df = ornek_df[X.columns]

# Örnek veri üzerinde tahmin yapma
ornek_tahmin = en_iyi_model.predict(ornek_df)

# Tahmin sonucunu yazdırma
if ornek_tahmin[0] == 7:
    print("Bu kişi yardım almamalı.")
elif ornek_tahmin[0] == 5:
    print("Bu kişi yardım almalı.")
else:
    print("Belirsiz bir durum söz konusu.")

