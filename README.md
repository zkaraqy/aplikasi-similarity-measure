# Aplikasi Similarity Measure dengan Angle Distance Signature

Aplikasi web untuk menganalisis similarity measure objek menggunakan metode angle distance signature. Aplikasi ini dapat melakukan segmentasi citra, perhitungan signature sudut, normalisasi, dan klasifikasi objek berdasarkan kesamaan bentuk.

## Fitur Utama

### 1. Load Citra
- Upload file citra dari komputer
- Pilih dari citra sample yang tersedia
- Mendukung format: PNG, JPG, JPEG, BMP, TIFF

### 2. Segmentasi
- Segmentasi otomatis menggunakan Otsu's thresholding
- Morphological operations untuk membersihkan hasil
- Menghasilkan citra biner untuk analisis lebih lanjut

### 3. Angle Signature
- Perhitungan jarak dari centroid ke boundary objek
- Analisis berdasarkan 360 sudut (0-359 derajat)
- Visualisasi dalam bentuk plot signature

### 4. Normalisasi
- Normalisasi signature ke range [0,1]
- Mengatasi variasi akibat scaling dan rotasi
- Plot signature yang telah dinormalisasi

### 5. Klasifikasi
- Perhitungan similarity measure menggunakan Euclidean distance
- Perbandingan dengan database referensi
- Menampilkan hasil klasifikasi dan tabel similarity

## Teknologi yang Digunakan

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Image Processing**: OpenCV, NumPy
- **Visualization**: Matplotlib
- **UI Framework**: Bootstrap 5.3.0

## Instalasi dan Setup

### 1. Persyaratan Sistem
- Python 3.7 atau lebih baru
- Pip (Python package installer)

### 2. Clone atau Download Project
```bash
git clone <repository-url>
cd aplikasi-similarity-measure
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Sample Images (Opsional)
```bash
python generate_samples.py
```

### 5. Jalankan Aplikasi
```bash
python app.py
```

Aplikasi akan berjalan pada `http://localhost:5000`

## Cara Penggunaan

### Langkah-langkah Analisis:

1. **Load Citra**
   - Pilih file citra atau gunakan sample yang tersedia
   - Klik tombol "Load Citra"
   - Citra akan tampil di panel "Citra Asal"

2. **Segmentasi**
   - Klik tombol "Segmentasi"
   - Hasil segmentasi akan tampil di panel "Citra Hasil"

3. **Angle Signature**
   - Klik tombol "Angle Signature"
   - Plot signature akan muncul di panel sebelah kanan
   - Menunjukkan jarak dari centroid untuk setiap sudut

4. **Normalisasi**
   - Klik tombol "Normalisasi"
   - Plot signature yang dinormalisasi akan diperbarui

5. **Klasifikasi**
   - Klik tombol "Klasifikasi"
   - Hasil klasifikasi akan muncul
   - Tabel similarity menampilkan perbandingan dengan objek referensi

## Struktur Project

```
aplikasi-similarity-measure/
├── app.py                    # Aplikasi Flask utama
├── image_processing.py       # Modul algoritma image processing
├── generate_samples.py       # Script untuk generate sample images
├── requirements.txt          # Dependencies Python
├── README.md                # Dokumentasi project
├── .github/
│   └── copilot-instructions.md
├── templates/
│   └── index.html           # Template HTML utama
├── static/
│   ├── css/                 # File CSS custom (kosong, menggunakan Bootstrap CDN)
│   ├── js/                  # File JavaScript custom (kosong, embedded di HTML)
│   ├── uploads/             # Folder untuk file yang diupload
│   ├── plots/               # Folder untuk plot yang dihasilkan
│   └── sample_images/       # Folder untuk citra sample
```

## API Endpoints

- `GET /` - Halaman utama aplikasi
- `POST /load_image` - Load citra dari file atau sample
- `POST /segment_image` - Segmentasi citra
- `POST /calculate_angle_signature` - Hitung angle signature
- `POST /normalize_signature` - Normalisasi signature
- `POST /classify_similarity` - Klasifikasi similarity
- `GET /get_file_list` - Ambil daftar file untuk tabel
- `GET /get_similarity_table` - Ambil data similarity untuk tabel

## Algoritma yang Diimplementasikan

### 1. Segmentasi Citra
- Konversi ke grayscale
- Otsu's automatic thresholding
- Morphological operations (closing, opening)
- Median blur untuk noise reduction

### 2. Angle Signature Calculation
- Deteksi kontur menggunakan `cv2.findContours()`
- Perhitungan centroid objek
- Perhitungan jarak maksimum untuk setiap sudut (0-359°)
- Interpolasi untuk sudut yang tidak memiliki titik boundary

### 3. Normalisasi
- Min-max normalization ke range [0,1]
- Mengatasi variasi ukuran objek
- Mempertahankan karakteristik bentuk relatif

### 4. Similarity Measure
- Euclidean distance antara signature
- Perbandingan dengan database referensi
- Ranking berdasarkan jarak terkecil

## Sample Images

Aplikasi menyediakan 5 sample images dengan bentuk berbeda:
1. `sample1_rectangle.png` - Bentuk persegi panjang
2. `sample2_circle.png` - Bentuk lingkaran
3. `sample3_triangle.png` - Bentuk segitiga
4. `sample4_ellipse.png` - Bentuk elips
5. `sample5_pentagon.png` - Bentuk pentagon

## Troubleshooting

### Error: Import tidak ditemukan
Pastikan semua dependencies sudah terinstall:
```bash
pip install -r requirements.txt
```

### Error: File tidak ditemukan
Pastikan struktur folder sesuai dan jalankan `generate_samples.py` untuk membuat sample images.

### Error: Port sudah digunakan
Ubah port di `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Ganti ke port lain
```

## Pengembangan Lebih Lanjut

### Fitur yang Dapat Ditambahkan:
1. **Multiple Object Detection** - Deteksi dan analisis multiple objek dalam satu citra
2. **Advanced Segmentation** - Implementasi algoritma segmentasi yang lebih canggih
3. **Database Management** - Sistem database untuk menyimpan signature referensi
4. **Batch Processing** - Pemrosesan multiple file sekaligus
5. **Export Results** - Export hasil analisis ke format CSV/Excel
6. **Real-time Processing** - Analisis real-time menggunakan webcam

### Algoritma Alternatif:
1. **Fourier Descriptors** - Sebagai alternatif angle signature
2. **Hu Moments** - Untuk shape analysis yang invariant
3. **Contour Matching** - Menggunakan `cv2.matchShapes()`
4. **Deep Learning** - CNN untuk feature extraction dan classification

## Kontribusi

Untuk berkontribusi pada project ini:
1. Fork repository
2. Buat branch baru untuk fitur (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -am 'Tambah fitur baru'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat Pull Request

## Lisensi

Project ini dibuat untuk tujuan edukasi dan penelitian.

## Kontak

Untuk pertanyaan atau saran, silakan buat issue di repository ini.