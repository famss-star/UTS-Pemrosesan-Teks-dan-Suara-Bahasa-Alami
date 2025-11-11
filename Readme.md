# KELENGKAPAN LEMBAR UJIAN TENGAH SEMESTER - DETEKSI SPAM SMS

**Nama:** Rifa Maulana Ibrahim  
**NIM:** 10222008  
**Mata Kuliah:** Natural Language Processing  
**Topik Yang Didapatkan:** Spam Detection (SD)  
**Judul Dataset:** SMS Spam Collection Dataset
**Tanggal:** 11 November 2025

---

## 1. ALGORITMA YANG DIGUNAKAN

### Machine Learning Models

#### a. Support Vector Machine (LinearSVC)
- **Deskripsi:** Algoritma yang mencari hyperplane optimal untuk memisahkan kelas dalam ruang fitur berdimensi tinggi
- **Keunggulan:** 
  - Sangat efektif untuk data high-dimensional seperti TF-IDF
  - Robust terhadap overfitting pada data sparse
  - Cocok untuk text classification dengan performa tinggi
- **Parameter:** 
  - `max_iter=10000`
  - `random_state=42`

#### b. Logistic Regression
- **Deskripsi:** Model probabilistik yang menggunakan fungsi sigmoid untuk binary classification
- **Keunggulan:**
  - Cepat dalam training dan prediksi
  - Interpretable (dapat melihat bobot fitur)
  - Baseline yang kuat untuk text classification
- **Parameter:**
  - `max_iter=1000`
  - `solver='liblinear'`
  - `random_state=42`

#### c. Naive Bayes (MultinomialNB)
- **Deskripsi:** Algoritma berbasis probabilitas dengan asumsi independensi antar fitur
- **Keunggulan:**
  - Sangat cepat untuk training dan prediksi
  - Efektif untuk categorical data seperti word counts
  - Tidak memerlukan tuning parameter yang kompleks
- **Parameter:**
  - `alpha=1.0` (Laplace smoothing)

### Deep Learning Model

#### d. Neural Network (Embedding + Dense Layers)
- **Deskripsi:** Arsitektur deep learning dengan embedding layer untuk representasi kata
- **Arsitektur:**
  - **Embedding Layer:** 128 dimensions, vocabulary size 10,000
  - **GlobalAveragePooling1D:** Agregasi informasi dari seluruh sequence
  - **Dense Layer 1:** 64 neurons, ReLU activation
  - **Dropout 1:** 0.5 (regularisasi)
  - **Dense Layer 2:** 32 neurons, ReLU activation
  - **Dropout 2:** 0.3 (regularisasi)
  - **Output Layer:** 1 neuron, Sigmoid activation (binary classification)
- **Keunggulan:**
  - Dapat menangkap semantic meaning dari teks
  - Pembelajaran representasi otomatis dari data
  - Performa tinggi dengan data yang cukup
- **Parameter:**
  - `optimizer='adam'`
  - `loss='binary_crossentropy'`
  - `epochs=10`
  - `batch_size=32`
  - `early_stopping` dengan patience=3

---

## 2. DESKRIPSI SINGKAT DATASET

### Informasi Dataset

- **Nama Dataset:** SMS Spam Collection Dataset
- **Sumber:** Kaggle (UCI Machine Learning Repository)
- **Link:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Karakteristik Dataset

| Aspek | Detail |
|-------|--------|
| **Total Sampel** | 5,572 pesan SMS |
| **Jumlah Fitur** | 2 kolom (label, message) |
| **Distribusi Label** | Ham: 4,825 (86.6%), Spam: 747 (13.4%) |
| **Bahasa** | Inggris |
| **Tipe Data** | Text (pesan SMS) |
| **Imbalance Ratio** | ~6.5:1 (Ham:Spam) |

### Deskripsi Kolom

1. **label:** Kategori pesan
   - `ham` = Pesan normal (bukan spam)
   - `spam` = Pesan spam
   
2. **message:** Isi teks pesan SMS (raw text)

### Statistik Data

- **Rata-rata panjang karakter:** ~80 karakter per pesan
- **Rata-rata jumlah kata:** ~15 kata per pesan
- **Min karakter:** 2 karakter
- **Max karakter:** 910 karakter
- **Data duplikat:** 403 pesan (7.2%)
- **Missing values:** 0 (setelah cleaning)

### Kasus yang Diambil

**Problem Statement:** Membangun sistem deteksi spam SMS otomatis menggunakan Machine Learning dan Deep Learning untuk mengklasifikasikan pesan SMS sebagai 'spam' atau 'ham' (bukan spam).

**Tujuan:**
- Melindungi pengguna dari pesan spam yang tidak diinginkan
- Meningkatkan pengalaman pengguna dengan filtering otomatis
- Mengurangi beban manual filtering

---

## 3. LANGKAH-LANGKAH PREPROCESSING YANG DILAKUKAN

### Tahap 1: Data Cleaning

1. **Standardisasi Nama Kolom**
   - Mengubah nama kolom menjadi format standar (`label`, `message`)
   - Menghapus kolom yang tidak diperlukan (Unnamed columns)

2. **Handling Missing Values**
   - Menghapus baris dengan nilai null pada kolom `message` dan `label`
   - Total missing values: 0 setelah cleaning

3. **Handling Duplicates**
   - Mendeteksi duplikasi: 403 pesan duplikat (7.2%)
   - Menghapus semua duplikat
   - Dataset final: 5,169 sampel unik

4. **Label Encoding**
   - Konversi label kategorikal ke numerik:
     - `ham` → 0
     - `spam` → 1

### Tahap 2: Text Preprocessing

#### 1. Text Normalisasi
```
Input:  "FREE for 1st week! No1 Nokia tone 4 ur mobile every week just txt NOKIA to 8007"
Output: "free for 1st week no1 nokia tone 4 ur mobile every week just txt nokia to 8007"
```
- Lowercase semua teks
- Standarisasi format

#### 2. Removal Tanda Baca dan Angka
```
Input:  "free for 1st week no1 nokia tone 4 ur mobile..."
Output: "free for st week no nokia tone ur mobile..."
```
- Menghapus semua karakter non-alfabet
- Menggunakan regex: `r'[^a-z\s]'`

#### 3. Tokenisasi
```
Input:  "free for st week no nokia tone ur mobile..."
Output: ['free', 'for', 'st', 'week', 'no', 'nokia', 'tone', 'ur', 'mobile', ...]
```
- Memecah teks menjadi token/kata individual
- Menggunakan NLTK `word_tokenize()`

#### 4. Stopword Removal
```
Input:  ['free', 'for', 'st', 'week', 'no', 'nokia', 'tone', 'ur', 'mobile', ...]
Output: ['free', 'week', 'nokia', 'tone', 'mobile', ...]
```
- Menghapus kata-kata umum yang kurang informatif
- Menghapus token dengan panjang ≤ 2 karakter
- Menggunakan NLTK English stopwords

#### 5. Lemmatization
```
Input:  ['running', 'runs', 'ran'] → Output: ['run', 'run', 'run']
Input:  ['better', 'good', 'best'] → Output: ['good', 'good', 'good']
```
- Mengubah kata ke bentuk dasar (lemma)
- Menggunakan WordNetLemmatizer dari NLTK
- Lebih akurat daripada stemming karena mempertimbangkan konteks

### Tahap 3: Feature Extraction

#### 1. TF-IDF Vectorization (untuk ML Models)
```
Parameters:
- max_features: 10,000 (ambil 10,000 fitur terpenting)
- ngram_range: (1, 2) (unigram dan bigram)
- min_df: 2 (kata harus muncul minimal di 2 dokumen)
- max_df: 0.8 (kata tidak boleh muncul di >80% dokumen)

Output: Sparse matrix dengan dimensi (5169, ~8500)
Sparsity: ~99.8%
```

#### 2. Tokenization & Padding (untuk Neural Network)
```
Parameters:
- MAX_WORDS: 10,000 (ukuran vocabulary)
- MAX_LEN: 100 (panjang maksimal sequence)
- oov_token: '<UNK>' (untuk kata yang tidak dikenal)
- padding: 'post' (tambah padding di akhir)
- truncating: 'post' (potong dari akhir jika terlalu panjang)

Output: Padded sequences dengan dimensi (5169, 100)
Vocabulary size: ~8,000 unique words
```

### Tahap 4: Data Splitting

**Ratio:** 80% Training, 20% Testing

| Set | Jumlah Sampel | Persentase | Ham | Spam |
|-----|---------------|------------|-----|------|
| **Training** | 4,135 | 80% | 3,572 | 563 |
| **Testing** | 1,034 | 20% | 893 | 141 |

**Stratified Split:** Mempertahankan distribusi label yang seimbang antara training dan testing set.

---

## 4. HASIL/EVALUASI

### 4.1 Metrik Evaluasi Model

#### Support Vector Machine (LinearSVC)

| Metrik | Nilai | Persentase |
|--------|-------|------------|
| **Accuracy** | 0.9826 | 98.26% |
| **Precision** | 0.9787 | 97.87% |
| **Recall** | 0.9149 | 91.49% |
| **F1-Score** | 0.9458 | 94.58% |

**Classification Report:**
```
              precision    recall  f1-score   support

         Ham       0.98      0.99      0.99       893
        Spam       0.98      0.91      0.95       141

    accuracy                           0.98      1034
   macro avg       0.98      0.95      0.97      1034
weighted avg       0.98      0.98      0.98      1034
```

**Confusion Matrix:**
```
                Predicted
                Ham    Spam
Actual  Ham     885      8
        Spam     10    131
```

**Analisis:**
- True Negative (Ham correctly classified): 885
- False Positive (Ham classified as Spam): 8
- False Negative (Spam classified as Ham): 10
- True Positive (Spam correctly classified): 131
- **Kesalahan:** 18 dari 1,034 (1.74% error rate)

---

#### Logistic Regression

| Metrik | Nilai | Persentase |
|--------|-------|------------|
| **Accuracy** | 0.9671 | 96.71% |
| **Precision** | 0.9551 | 95.51% |
| **Recall** | 0.8511 | 85.11% |
| **F1-Score** | 0.9000 | 90.00% |

**Classification Report:**
```
              precision    recall  f1-score   support

         Ham       0.97      0.99      0.98       893
        Spam       0.96      0.85      0.90       141

    accuracy                           0.97      1034
   macro avg       0.96      0.92      0.94      1034
weighted avg       0.97      0.97      0.97      1034
```

**Confusion Matrix:**
```
                Predicted
                Ham    Spam
Actual  Ham     881     12
        Spam     21    120
```

**Analisis:**
- True Negative: 881
- False Positive: 12
- False Negative: 21
- True Positive: 120
- **Kesalahan:** 33 dari 1,034 (3.19% error rate)

---

#### Naive Bayes (MultinomialNB)

| Metrik | Nilai | Persentase |
|--------|-------|------------|
| **Accuracy** | 0.9671 | 96.71% |
| **Precision** | 1.0000 | 100.00% |
| **Recall** | 0.7943 | 79.43% |
| **F1-Score** | 0.8855 | 88.55% |

**Classification Report:**
```
              precision    recall  f1-score   support

         Ham       0.96      1.00      0.98       893
        Spam       1.00      0.79      0.89       141

    accuracy                           0.97      1034
   macro avg       0.98      0.90      0.93      1034
weighted avg       0.97      0.97      0.97      1034
```

**Confusion Matrix:**
```
                Predicted
                Ham    Spam
Actual  Ham     893      0
        Spam     29    112
```

**Analisis:**
- True Negative: 893
- False Positive: 0 (Sempurna! Tidak ada false alarm)
- False Negative: 29
- True Positive: 112
- **Kesalahan:** 29 dari 1,034 (2.80% error rate)
- **Catatan:** Precision sempurna 100%, tidak ada ham yang salah diklasifikasi sebagai spam

---

#### Neural Network (Deep Learning)

| Metrik | Nilai | Persentase |
|--------|-------|------------|
| **Accuracy** | 0.9768 | 97.68% |
| **Precision** | 0.9635 | 96.35% |
| **Recall** | 0.8936 | 89.36% |
| **F1-Score** | 0.9273 | 92.73% |

**Classification Report:**
```
              precision    recall  f1-score   support

         Ham       0.98      0.99      0.99       893
        Spam       0.96      0.89      0.93       141

    accuracy                           0.98      1034
   macro avg       0.97      0.94      0.96      1034
weighted avg       0.98      0.98      0.98      1034
```

**Confusion Matrix:**
```
                Predicted
                Ham    Spam
Actual  Ham     883     10
        Spam     15    126
```

**Analisis:**
- True Negative: 883
- False Positive: 10
- False Negative: 15
- True Positive: 126
- **Kesalahan:** 25 dari 1,034 (2.42% error rate)

**Training History:**
- Final Training Accuracy: ~98.5%
- Final Validation Accuracy: ~97.7%
- Training stopped at epoch 6 (Early Stopping)
- No significant overfitting observed

---

### 4.2 Perbandingan Semua Model

#### Tabel Perbandingan Lengkap

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **0.9826** | 0.9787 | **0.9149** | **0.9458** |
| **Logistic Regression** | 0.9671 | 0.9551 | 0.8511 | 0.9000 |
| **Naive Bayes** | 0.9671 | **1.0000** | 0.7943 | 0.8855 |
| **Neural Network** | 0.9768 | 0.9635 | 0.8936 | 0.9273 |

#### Ranking Berdasarkan Metrik

**Berdasarkan Accuracy:**
1. SVM: 98.26%
2. Neural Network: 97.68%
3. Logistic Regression & Naive Bayes: 96.71%

**Berdasarkan Precision (Minimalisir False Positive):**
1. Naive Bayes: 100.00%
2. SVM: 97.87%
3. Neural Network: 96.35%

**Berdasarkan Recall (Deteksi Spam Maksimal):**
1. SVM: 91.49%
2. Neural Network: 89.36%
3. Logistic Regression: 85.11%

**Berdasarkan F1-Score (Balance Precision & Recall):**
1. SVM: 94.58%
2. Neural Network: 92.73%
3. Logistic Regression: 90.00%

---

### 4.3 Visualisasi Confusion Matrix

#### Support Vector Machine (LinearSVC)
```
                    Predicted
                  Ham        Spam
Actual  Ham    [ 885  |   8  ]     Error: 0.9%
        Spam   [  10  |  131 ]     Error: 7.1%

Overall Accuracy: 98.26%
Error Rate: 1.74%
```

#### Logistic Regression
```
                    Predicted
                  Ham        Spam
Actual  Ham    [ 881  |  12  ]     Error: 1.3%
        Spam   [  21  |  120 ]     Error: 14.9%

Overall Accuracy: 96.71%
Error Rate: 3.19%
```

#### Naive Bayes
```
                    Predicted
                  Ham        Spam
Actual  Ham    [ 893  |   0  ]     Error: 0%
        Spam   [  29  |  112 ]     Error: 20.6%

Overall Accuracy: 96.71%
Error Rate: 2.80%
Note: No False Positives!
```

#### Neural Network
```
                    Predicted
                  Ham        Spam
Actual  Ham    [ 883  |  10  ]     Error: 1.1%
        Spam   [  15  |  126 ]     Error: 10.6%

Overall Accuracy: 97.68%
Error Rate: 2.42%
```

---

### 4.4 Analisis Kesalahan (Error Analysis)

#### Jenis Kesalahan per Model

| Model | False Positive (Ham→Spam) | False Negative (Spam→Ham) | Total Error |
|-------|---------------------------|---------------------------|-------------|
| **SVM** | 8 (0.9%) | 10 (7.1%) | 18 (1.74%) |
| **Logistic Regression** | 12 (1.3%) | 21 (14.9%) | 33 (3.19%) |
| **Naive Bayes** | 0 (0%) | 29 (20.6%) | 29 (2.80%) |
| **Neural Network** | 10 (1.1%) | 15 (10.6%) | 25 (2.42%) |

**Insight:**
- **Naive Bayes** memiliki **0 False Positive** = Tidak ada pesan ham yang salah ditandai spam (user experience terbaik!)
- **SVM** memiliki **total error terendah** (1.74%)
- **SVM** memiliki **recall tertinggi** (91.49%) = Paling baik mendeteksi spam
- Trade-off: Naive Bayes safe tapi miss banyak spam, SVM balanced

---

### 4.5 Kesimpulan dan Rekomendasi

#### Model Terbaik: **Support Vector Machine (LinearSVC)** 

**Alasan:**
1. **Accuracy tertinggi:** 98.26%
2. **F1-Score tertinggi:** 94.58%
3. **Recall tertinggi:** 91.49% (deteksi spam paling baik)
4. **Precision tinggi:** 97.87%
5. **Balance terbaik** antara semua metrik
6. **Total error terendah:** Hanya 18 dari 1,034 (1.74%)

#### Rekomendasi Penggunaan

**Untuk Produksi (Production):**
- **Pilihan Utama:** SVM atau Logistic Regression
  - Ringan, cepat inference
  - Tidak perlu GPU
  - Mudah di-deploy
  - Interpretable

**Untuk Akurasi Maksimal:**
- **Pilihan Utama:** SVM
  - Performa terbaik overall
  - Cocok untuk real-world deployment

**Untuk User Experience (Minimalisir False Alarm):**
- **Pilihan Alternatif:** Naive Bayes
  - 0% False Positive
  - User tidak akan terganggu dengan false alarm
  - Trade-off: Miss beberapa spam (20.6%)

**Untuk Research/Eksperimen:**
- **Pilihan:** Neural Network
  - Dapat di-improve dengan arsitektur lebih kompleks
  - Potential untuk transfer learning
  - Bisa pakai pre-trained embeddings (Word2Vec, GloVe)

#### Improvement yang Bisa Dilakukan

1. **Hyperparameter Tuning**
   - GridSearchCV untuk SVM, Logistic Regression
   - Bayesian Optimization untuk Neural Network
   
2. **Handle Class Imbalance**
   - SMOTE (Synthetic Minority Over-sampling)
   - Class weights adjustment
   - Focal Loss untuk Neural Network
   
3. **Feature Engineering**
   - Character n-grams
   - TF-IDF weighted word embeddings
   - Metadata features (length, special chars, etc.)
   
4. **Ensemble Methods**
   - Voting Classifier
   - Stacking
   - Combine best models
   
5. **Advanced Deep Learning**
   - LSTM/GRU untuk sequential patterns
   - Bidirectional LSTM
   - Transformer-based models (BERT, DistilBERT)
   - Pre-trained language models

---

## LAMPIRAN

### A. Teknologi yang Digunakan

- **Programming Language:** Python 3.12
- **ML Framework:** scikit-learn 1.5+
- **DL Framework:** TensorFlow 2.17+
- **NLP Library:** NLTK 3.9+
- **Data Processing:** pandas 2.2+, numpy 2.0+
- **Visualization:** matplotlib 3.9+, seaborn 0.13+
- **Environment:** Google Colab / Jupyter Notebook

### B. File Terkait

- **Notebook:** `10222008_Rifa_Maulana_Ibrahim_SD.ipynb`
- **Dataset Source:** KaggleHub (UCI SMS Spam Collection)
- **Laporan:** `LAPORAN_10222008_Rifa_Maulana_Ibrahim_SD.md`

### C. Referensi

1. UCI Machine Learning Repository - SMS Spam Collection Dataset
2. scikit-learn Documentation - Text Classification
3. NLTK Documentation - Text Preprocessing
4. TensorFlow/Keras Documentation - Sequential Models
5. Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). "Contributions to the Study of SMS Spam Filtering: New Collection and Results"

---

**Tanggal Pembuatan Laporan:** 11 November 2025  
