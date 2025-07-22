# 🩺 Prediksi Risiko Diabetes Menggunakan Machine Learning

> Proyek ini bertujuan untuk memprediksi risiko seseorang terkena diabetes menggunakan algoritma machine learning berbasis data kesehatan pasien perempuan keturunan Pima Indian.

## 📌 Deskripsi Proyek
Diabetes merupakan penyakit kronis yang sering tidak terdiagnosis sejak dini. Dengan pendekatan machine learning, proyek ini mencoba membangun model prediksi risiko diabetes berdasarkan fitur-fitur seperti kadar glukosa, tekanan darah, usia, dan indeks massa tubuh.

## 🎯 Tujuan
- Membangun model klasifikasi untuk memprediksi apakah seseorang menderita diabetes.
- Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap prediksi risiko.
- Membandingkan performa beberapa algoritma: Logistic Regression, Decision Tree, dan Random Forest.

## 📁 Dataset
- Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Total data: 768 baris, 9 kolom
- Fitur penting: `Glucose`, `BMI`, `Age`, `DiabetesPedigreeFunction`, dll.
- Target: `Outcome` (1 = diabetes, 0 = tidak)

## 🛠️ Tools & Library
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## ⚙️ Proses
1. **Data Cleaning & Preprocessing**
   - Mengganti nilai 0 tidak valid dengan median
   - Standarisasi fitur numerik
   - Train-test split (80:20)

2. **Modeling**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Evaluasi dengan: Accuracy, Precision, Recall, F1-Score

3. **Hyperparameter Tuning**
   - Menggunakan GridSearchCV untuk Random Forest

4. **Feature Importance**
   - Fitur paling berpengaruh: `Glucose`, `BMI`, `Pedigree`, `Age`

## 📊 Hasil Evaluasi Model Terbaik (Random Forest)
| Metrik     | Skor   |
|------------|--------|
| Accuracy   | 78%    |
| Precision  | 72%    |
| Recall     | 61%    |
| F1-Score   | 66%    |

> Random Forest dipilih sebagai model terbaik karena memberikan keseimbangan performa pada semua metrik serta kemampuan menangani data kompleks.

## 🔍 Insight Utama
- **Glucose** memiliki pengaruh paling besar terhadap prediksi risiko diabetes.
- Model machine learning dapat membantu skrining dini untuk kasus diabetes di komunitas medis.

## 📎 Referensi
- Kaggle Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Studi pendukung dari jurnal dan penelitian lain yang relevan (lihat laporan lengkap).

---

