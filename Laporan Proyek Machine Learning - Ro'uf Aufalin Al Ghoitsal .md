# Laporan Proyek Machine Learning - Rouf Aufalin Al Ghoitsal

## Kesehatan - Predictive Analytics : Diabetes 
Diabetes merupakan salah penyakit tidak menular yang menjadi penyebab kematian terbanyak di dunia. Indonesia merupakan salah satu dari 38 negara yang tergabung dalam Organisasi Internasional Diabetes Federation (IDF) [1]. Menurut IDF indonesia memiliki penderita diabetes ke 5 tertinggi didunia terdapat 20,4 juta kasus pada usia 20-79 tahun di tahun 2024. 

Diabetes merupakan penyakit dimana kondisi kandungan gula yang terdapat dalam darah tidak dapat diolah dengan baik oleh tubuh [2]. Penyakit ini dikenal sebagai silent killer dikarenakan para penderita tidak menyadari dan saat diketahui sudah terjadi banyak komplikasi. Dari beberapa penelitian sebelumnya [3] menjelaskan bahwa kebanyakan masyarakat terkena diabetes karena pola hidup yang tidak teratur dan konsumsi makanan yang mengandung gula secara berlebihan.

Keterlambatan diagnosis penyakit diabetes menjadi salah satu penyebab meningkatnya jumlah penderita diabetes. Untuk mengendalikan peningkatan ini diperlukan diagnosis penyakit tersebut secara dini untuk mencegah komplikasi [4]. Namun dengan perkembangan teknologi yang pesat ini dapat digunakan untuk membantu dokter dalam memprediksi resiko penyakit tersebut. 

Salah satu pendekatan teknologi yang potensial adalah machine learning, yang dapat digunakan untuk menganalisis pola dari data historis guna memprediksi risiko diabetes. Dari berbagai algoritma yang tersedia, metode klasifikasi merupakan salah satu yang umum digunakan untuk memprediksi apakah seseorang berisiko menderita diabetes berdasarkan atribut-atribut tertentu.

Klasifikasi adalah sebuah proses untuk menemukan model atau fungsi dengan mengelompokan kelas data dengan tujuan untuk memperkirakan kelas dari suatu objek yang labelnya tidak diketahui.  Menurut hasil penelitian dari [1] [4] metode klasifikasi mampu digunakan untuk memprediksi risiko penyakit diabetes. Hal ini diharapkan mampu dimanfaatkan untuk mendiagnosa penyakit diabetes lebih awal.

[1]	W. Sohibul, A. Hadiana, dan F. Umbara, “Prediksi Penyakit Diabetes Menggunakan Algoritma Support Vector Machine (SVM),” 2022. [Daring]. Tersedia pada: https://e-journal.unper.ac.id/index.php/informatics

[2]	N. Hackworth dkk., “A Risk Factor Profile for Pre-diabetes: Biochemical, Behavioural, Psychosocial and Cultural Factors,” E-Journal of Applied Psychology, vol. 3, no. 2, Des 2007, doi: 10.7790/ejap.v3i2.89.

[3]	P. Piko, N. A. Werissa, S. Fiatal, J. Sandor, dan R. Adany, “Impact of Genetic Factors on the Age of Onset for Type 2 Diabetes Mellitus in Addition to the Conventional Risk Factors,” J Pers Med, vol. 11, no. 1, hlm. 6, Des 2020, doi: 10.3390/jpm11010006.

[4]	J. J. Purnama, S. Rahayu, S. Nurdiani, T. Haryanti, dan N. A. Mayangky, “Analisis Algoritma Klasifikasi Neural Network Untuk Diagnosis Penyakit Diabetes,” IJCIT (Indonesian Journal on Computer and Information Technology), vol. 5, no. 1, Mei 2020, doi: 10.31294/ijcit.v5i1.6391.

## Business Understanding

### Problem Statements
- Banyak penderita diabetes di Indonesia tidak terdiagnosis sejak dini sehingga menimbulkan risiko komplikasi serius
- Perlu dilakukan identifikasi faktor-faktor gaya hidup dan kesehatan yang paling berpengaruh terhadap risiko diabetes, guna mendukung edukasi dan pencegahan dini.

### Goals
- Mengembangkan model prediksi berbasis machine learning yang dapat digunakan untuk mendeteksi risiko diabetes sejak dini.
- Mengidentifikasi fitur-fitur (variabel input) yang paling signifikan dalam mempengaruhi hasil prediksi risiko diabetes.

### Solution Staments
- Menggunakan algoritma klasifikasi  Logistic Regression, Decision Tree, dan Random Forest untuk memodelkan risiko diabetes. Perbandingan performa masing-masing algoritma akan dilakukan untuk menentukan model terbaik.
- Menerapkan feature importance analysis dari Random Forest untuk mengevaluasi pengaruh setiap variabel input terhadap prediksi model.
- Model dievaluasi menggunakan metrik evaluasi yaitu accuracy, precision, recall dan F1 score


## Data Understanding
Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Data ini memiliki total 768 baris dan 9 kolom. Data ini memiliki beberapa nilai nol yang tidak masuk akal pada beberapa fitur.

Tujuan dari kumpulan data ini adalah untuk memprediksi secara diagnostik apakah pasien menderita diabetes atau tidak, berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Beberapa batasan ditempatkan pada pemilihan contoh-contoh ini dari basis data yang lebih besar. 

Secara khusus, semua pasien di sini adalah perempuan berusia minimal 21 tahun yang berasal dari suku Indian Pima. [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).


### Variabel-variabel pada Pima Indians Diabetes Database adalah sebagai berikut:
- Pregnancies : merupakan angka berapa kali hami
- Glucose : merupakan Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- BloodPressure : merupakan tekanan darah diastolik (mmHg)
- SkinThickness : Merupakan ketebalan lipatan kulit trisep (mm)
- Insulin = merupakan kadar 2-Hour serum insulin (mu U/ml)
- BMI : merupakan indeks masa tubuh yang diukur dengan rumus (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction :  merupakan fungsi yang menghasilkan nilai pengaruh riwayat penyakit diabetes pada seseorang.
- Age : Usia
- Outcome : merupakan variabel target dimana 1 adalah menderita dan 0 tidak

### Exploratory Data Analysis (EDA)
- Univariate Analysis tiap variabel
- Bivariate Analysis dengan variabel target
- Mencari korelasi fitur menggunakan Correlation Matrix

## Data Preparation
Tahap ini dilakukan untuk mempersiapkan data sebelum lanjut ke tahap modeling.
### Load Dataset
Dataset yang digunakan adalah Pima Indian Diabetes Dataset yang berisi informasi kesehatan pasien perempuan keturunan Pima Indian berusia 21 tahun ke atas. Dataset terdiri dari 768 entri dan 8 fitur independen serta 1 label target (Outcome).

### Data Cleaning 
Langkah yang dilakukan:
- Mengecek nilai nol (zero) pada kolom yang tidak mungkin bernilai nol, seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI.
- Mengganti nilai nol pada kolom tersebut dengan nilai median kolom masing-masing.

Dalam dataset ini, nilai nol di kolom-kolom tersebut dianggap sebagai nilai yang hilang (missing values) karena secara medis nilai nol pada tekanan darah, kadar glukosa, atau BMI tidak masuk akal.

### Data Splitting
Langkah yang dilakukan:
- Membagi data menjadi data latih (train) dan data uji (test) dengan rasio 80:20 menggunakan fungsi train_test_split() dari scikit-learn.

Pemecahan ini bertujuan untuk melatih model menggunakan sebagian besar data dan menguji performa model terhadap data yang belum pernah dilihat.

### Standarisasi
Langkah yang dilakukan: 
- Mengubah distribusi data dengan nilai rata-rata dan standar deviase 1

Hal ini dilakukan supaya algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. 

## Modeling
Pada tahap ini dilakukan proses pembangunan model machine learning menggunakan data yang telah melalui tahap preparation. Beberapa algoritma klasifikasi digunakan untuk membandingkan performa dan memilih model terbaik dalam memprediksi risiko diabetes.

### 1. Model yang Digunakan

#### a. Logistic Regression
- **Cara Kerja:** Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memodelkan probabilitas suatu kelas. Model ini menggunakan fungsi sigmoid (logistic function) untuk mengubah output linear menjadi nilai probabilitas antara 0 dan 1. Model ini mencoba menemukan garis (atau bidang dalam dimensi lebih tinggi) pemisah terbaik antara kelas dengan memaksimalkan likelihood dari parameter.

- **Parameter** 
default dari pustaka scikit-learn, yaitu:
    - penalty='l2': regulasi L2 untuk menghindari overfitting,
    - solver='lbfgs': solver yang cocok untuk dataset kecil sampai menengah.
- **Kelebihan**:
  - Sederhana dan cepat dilatih.
  - Memberikan output probabilistik yang mudah diinterpretasikan.
- **Kekurangan**:
  - Tidak mampu menangkap hubungan non-linear yang kompleks.

#### b. Decision Tree
- **Cara Kerja:** Decision Tree bekerja dengan membagi dataset menjadi beberapa subset berdasarkan fitur-fitur tertentu menggunakan metrik seperti Gini Impurity atau Entropy. Model ini membentuk struktur seperti pohon, di mana setiap node berisi keputusan berdasarkan nilai fitur, dan daun berisi prediksi kelas.
- **Parameter** default dari pustaka scikit-learn, yaitu:
    - criterion='gini': menggunakan indeks Gini untuk mengukur kualitas split,
    - max_depth=None: pohon akan tumbuh hingga semua daun bersifat murni (pure) atau memiliki jumlah sampel kurang dari min_samples_split.
- **Kelebihan**:
    - Mudah diinterpretasikan.
    - Dapat menangani data kategorikal dan numerik.
- **Kekurangan**:
  - Cenderung overfitting jika tidak di-pruning dengan baik.

#### c. Random Forest
- **Cara Kerja:** Random Forest adalah metode ensemble yang membentuk banyak Decision Tree dan menggabungkan hasil prediksi dari masing-masing pohon menggunakan voting mayoritas (untuk klasifikasi). Setiap pohon dilatih pada subset acak dari data dan fitur (bagging), sehingga menghasilkan model yang lebih stabil dan mengurangi overfitting.

- **Parameter** yang digunakan default dari pustaka scikit-learn
    - n_estimators=100: jumlah pohon dalam hutan
    - criterion='gini': metrik untuk kualitas split
    - max_depth=None: setiap pohon akan tumbuh tanpa batas kecuali dihentikan oleh parameter lain.
- **Kelebihan**:
  - Akurasi tinggi dan lebih tahan terhadap overfitting dibanding decision tree tunggal.
  - Dapat menangani fitur dalam jumlah besar dan menangani missing value secara moderat.
- **Kekurangan**:
  - Kurang dapat diinterpretasikan.
  - Lebih lambat dalam training dan prediksi dibanding model 

### 2. Evaluasi Model

Setiap model diuji menggunakan data uji (20% dari dataset) dan dievaluasi menggunakan metrik:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### 3. Pemilihan Model Terbaik

Berdasarkan hasil evaluasi, **Random Forest** dipilih sebagai model terbaik karena memiliki nilai tertinggi pada hampir semua metrik yang menunjukkan kemampuan membedakan antara penderita dan non-penderita diabetes.
Random Forest memberikan hasil terbaik karena merupakan algoritma ensemble yang menggabungkan banyak decision tree, sehingga mampu menangkap pola kompleks, mengurangi overfitting, dan menghasilkan generalisasi yang lebih baik pada data uji. Dibandingkan Logistic Regression dan Decision Tree, Random Forest lebih fleksibel dalam menangani data non-linear, lebih stabil, serta memiliki performa evaluasi yang seimbang antara akurasi, precision, recall, dan F1-score, menjadikannya pilihan optimal untuk kasus klasifikasi diabetes ini.

### 4. Hyperparameter Tuning (Improvement)

Untuk meningkatkan performa lebih lanjut, dilakukan **tuning hyperparameter** pada Random Forest menggunakan **GridSearchCV**, dengan parameter berikut:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

### Metrik Evaluasi yang Digunakan

Karena proyek ini merupakan kasus **klasifikasi biner** (penderita diabetes atau tidak), maka metrik evaluasi yang digunakan adalah sebagai berikut:

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

**Precision** = TP / (TP + FP)

**Recall** = TP / (TP + FN)

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

### Hasil Evaluasi Model

Model terbaik yang dipilih adalah **Random Forest**. Berikut adalah hasil evaluasi pada data uji:

| Metrik            | Nilai   |
|-------------------|---------|
| Accuracy          | 0.78    |
| Precision         | 0.72    |
| Recall            | 0.61    |
| F1-Score          | 0.66    |

- Akurasi (77.92%) menunjukkan bahwa sekitar 78% prediksi model adalah benar, yang berarti model cukup baik secara umum dalam memprediksi diabetes.

- Precision (71.74%) mengindikasikan bahwa ketika model memprediksi seseorang menderita diabetes, sekitar 72% prediksi tersebut benar. Ini cukup baik, tetapi ada sekitar 28% prediksi positif yang salah (false positives).

- Recall (61.11%) berarti model berhasil mendeteksi sekitar 61% dari semua penderita diabetes yang sebenarnya. Ini menunjukkan masih ada 39% penderita yang tidak terdeteksi (false negatives), yang perlu diperbaiki terutama dalam konteks diagnosis kesehatan.

- F1-Score (66.00%) mengindikasikan keseimbangan antara precision dan recall, menunjukkan performa model yang moderat dalam menangani trade-off antara false positives dan false negatives.

### Fitur Importance Menggunakan Random Forest
Dari hasil perhitungan feature importance, fitur-fitur diurutkan berdasarkan kontribusinya terhadap prediksi diabetes sebagai berikut:

**Glucose (0.274)**
Fitur ini memiliki kontribusi terbesar (sekitar 27,4%) dalam menentukan risiko diabetes. Hal ini masuk akal karena kadar glukosa darah adalah indikator langsung dari diabetes.
**BMI (Body Mass Index) (0.162)**
Indeks massa tubuh juga berperan penting (16,2%) karena obesitas atau kelebihan berat badan merupakan faktor risiko utama diabetes tipe 2.
**DiabetesPedigreeFunction (0.125)**
Fitur ini mengukur riwayat keluarga diabetes, yang berarti faktor genetik juga cukup berpengaruh (12,5%).
**Age (0.113)**
Usia memberikan kontribusi signifikan (11,3%), karena risiko diabetes meningkat seiring bertambahnya usia.
**Insulin (0.091)**
Level insulin dalam darah juga berpengaruh (9,1%), berkaitan dengan fungsi pankreas dan metabolisme.
**BloodPressure (0.084)**
Tekanan darah memiliki pengaruh (8,4%), karena hipertensi sering terkait dengan diabetes dan kondisi metabolik lain.
**Pregnancies (0.081)**
Jumlah kehamilan turut berperan (8,1%), yang bisa memengaruhi risiko diabetes gestasional dan risiko diabetes secara umum.
**SkinThickness (0.070)**
Ketebalan kulit yang diukur untuk memperkirakan kadar lemak juga memberi kontribusi (7,0%), walau relatif kecil dibanding fitur lain.