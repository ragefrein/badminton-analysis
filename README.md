# Rally Vision

Aplikasi desktop Windows untuk tracking pemain badminton single dan doubles. Antarmuka native PySide6, inferensi lokal YOLO11s Pose + OSNet ReID, dukungan GPU NVIDIA / fallback CPU. Fase ini berfokus pada identitas pemain; klasifikasi pukulan belum dibuat.

## Jalankan

Di komputer ini dependensi dan dua model sudah tersedia. Klik dua kali **Run Rally Vision.bat**, atau jalankan dari folder project:

```powershell
python main.py
python main.py --video single.mp4
```

Untuk instalasi baru dengan Python 3.12:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
# CPU saja:
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Cpu
```

Script membuat `.venv`, memasang dependensi, dan mengambil bobot publik jika belum ada. Launcher memilih `.venv` jika tersedia. Ini aplikasi desktop berbasis Python; belum dikemas sebagai executable mandiri.

## Penggunaan

1. Pilih Doubles / Singles atau buka video sendiri. Dua rekaman bawaan dibuka pada detik 30 agar langsung menampilkan lapangan. Slider dapat digunakan untuk mencari waktu lain.
2. Isi nama setiap pemain di roster kanan, lalu tekan Tab atau klik di luar kolom. Nama tersimpan otomatis secara lokal untuk video tersebut.
3. **Periksa usulan ID awal.** P01/P02 adalah urutan kiri/kanan pada sisi jauh saat inisialisasi doubles, P03/P04 pada sisi dekat. Setelah dikunci, ID tidak mengikuti urutan kiri/kanan lagi. Untuk singles hanya P01/P02.
4. Untuk mengoreksi, klik **Pilih** pada ID yang diinginkan, tunggu video dijeda, lalu klik tubuh pemain di video. Jika tubuh sudah ditempati ID lain, kedua state fisiknya ditukar; nama tetap melekat pada ID roster. Deteksi kandidat ditandai D1, D2, dan seterusnya saat memilih.
5. Klik **Mulai**. Space menjeda/melanjutkan; tombol panah kanan memproses satu frame saat dijeda. Input nama tetap mendukung spasi.
6. Menu **Opsi** berisi pengaturan skeleton, lingkaran kaki dengan efek perspektif 3D, dan jejak gerak. Tampilan utama hanya berisi video, input nama, dan kontrol pemutaran.
7. Gunakan **Opsi → Kalibrasi lapangan** bila kamera/video berbeda. Klik sudut garis luar lapangan ganda dalam urutan kiri jauh → kanan jauh → kanan dekat → kiri dekat. Preset bawaan disesuaikan dengan dua video contoh. Lapangan fisik yang dipakai homografi adalah 6.1 × 13.4 m, termasuk saat pertandingan singles.
8. **Ekspor** menghasilkan `tracking.csv` dan `session.json`. **Opsi → Simpan frame** menyimpan gambar video dengan nama dan overlay ke `artifacts/`.

## Bagaimana identitas dijaga

Roster berisi tepat 2 atau 4 ID dan tidak menambahkan ID otomatis. Setiap frame diproses dengan deteksi pose, embedding penampilan OSNet, prediksi gerak pinggul dan kepala, batas perpindahan sesuai ukuran tubuh, dan pembatasan sisi lapangan. Posisi kepala membantu membedakan pemain ketika pose pinggul bertumpuk; gerakan pergelangan kaki tidak lagi menentukan identitas. Assignment dilakukan secara global dengan Hungarian matching dan dibandingkan dengan assignment alternatif.

Deteksi mempertahankan kandidat hingga confidence 0.12 dan memakai batas overlap 0.72 agar pemain bertumpuk tidak terlalu cepat terhapus. Jika kandidat kuat pada salah satu sisi kurang dari roster, model memeriksa ulang crop sisi lapangan tersebut dengan skala lebih besar. Duplikasi digabung berdasarkan kesamaan pose; dua kepala yang jelas berbeda tetap dianggap dua pemain. Kandidat confidence rendah hanya boleh melanjutkan ID yang baru terlihat, tidak membentuk ID awal atau memulihkan ID yang hilang lama. Gallery tampilan tidak belajar dari kandidat confidence rendah. Setelah kehilangan panjang, pemulihan dengan kemiripan tampilan sedang memerlukan tiga frame konsisten.

Jika pencocokan ambigu atau pemain tidak terlihat, status menjadi **Perlu cek / Tidak terlihat**. Skeleton, nama, dan lingkaran untuk identitas itu ditahan; koordinat lama tidak ditampilkan seolah-olah hasil deteksi baru. Gallery tampilan tidak diperbarui ketika tubuh bertumpuk atau pencocokan kurang kuat. Identitas dapat pulih ketika buktinya cukup jelas; koreksi manual tetap tersedia.

**Tidak ada jaminan nol pertukaran ID pada video arbitrer**, khususnya saat rekan satu tim memakai pakaian yang sama dan tertutup total. Belum ada anotasi ground truth untuk mengukur IDF1/HOTA atau jumlah ID switch pada seluruh rekaman. Usulan awal harus diverifikasi pengguna. Pergantian kamera yang terdeteksi menjeda tracking dan mewajibkan pemasangan ulang. Seek dan perubahan kalibrasi juga membatalkan posisi sebelumnya dan memulai sesi statistik baru, sambil mempertahankan nama. Pertukaran sisi lapangan harus diikuti pemasangan ID ulang.

Pipeline mengasumsikan kamera lapangan tetap. Pergantian kamera yang gradual/zoom atau potongan dengan tampilan sangat mirip bisa lolos detektor cut sederhana; jeda dan lakukan kalibrasi/pemasangan ulang bila perlu. CPU lebih lambat; aplikasi memproses setiap frame, sehingga kecepatan pemutaran mengikuti kemampuan inferensi, tanpa melompati frame untuk mengejar waktu nyata.

## Data dan statistik

- `sessions/<video-hash>.json`: nama roster, mode, dan kalibrasi per video.
- `sessions/run-*.sqlite`: log inkremental dari frame yang benar-benar dianalisis dan peristiwa koreksi/cut.
- `tracking.csv`: frame sumber, ID roster, waktu sumber, nama, status, skor pencocokan (bukan probabilitas identitas yang terkalibrasi), posisi lapangan, estimasi jarak kumulatif, 17 keypoint `(x,y,confidence)` dengan koordinat gambar ternormalisasi, asal deteksi (`full` / `crop`), dan confidence detektor.
- ID yang tidak teramati tetap memiliki baris dengan posisi/keypoint kosong. Tidak ada interpolasi skeleton palsu. Input nama terbaru diterapkan juga ke baris ekspor sebelumnya.
- `session.json`: metadata model, roster, kalibrasi, jumlah frame yang diproses, status pengamatan, estimasi jarak, dan peristiwa manual.

Jarak merupakan estimasi dari titik kaki dan homografi bidang lapangan. Gerakan kaki, lompatan, jitter keypoint, dan kesalahan kalibrasi memengaruhi hasil. Jarak tidak ditambahkan melintasi kehilangan panjang atau perpindahan yang tidak masuk akal. Statistik berlaku untuk segmen/frame yang diproses, bukan otomatis seluruh pertandingan.

## Verifikasi

```powershell
python -m pytest -q
python scripts/verify_desktop.py
python analyze.py doubles.mp4 --start 30 --seconds 8 --output artifacts/doubles-smoke
python analyze.py single.mp4 --mode singles --start 30 --seconds 8 --output artifacts/singles-smoke
```

`analyze.py` menginisialisasi usulan kiri/kanan secara otomatis hanya untuk pengujian, mengekspor klip MP4 dengan ID numerik, CSV/JSON, gambar contoh, dan laporan. Klip pengujian tersebut bukan bukti identitas benar tanpa ground truth. Viewer desktop dan simpan frame mendukung nama Unicode melalui Qt.

Pengujian logika mencakup crossing, urutan detektor berubah, penahanan identitas ambigu, pemain hilang dan muncul kembali, larangan mencocokkan lawan di sisi lain, koreksi manual, seek/cut, kalibrasi, dan ekspor nama Unicode. Integrasi desktop menguji inferensi nyata pada kedua video, input nama, pemilihan tubuh, play/pause, seek, dan ekspor.

Perbandingan versi 0.2 pada rekaman bawaan:

| Segmen | Frame | Frame dengan ID tidak lengkap: sebelumnya | Versi 0.2 |
| --- | ---: | ---: | ---: |
| Doubles, 00:30–00:38 | 240 | 24 | 5 |
| Doubles, 00:38–00:58 | 600 | 27 | 0 |

Ini pengukuran kontinuitas ID pada cuplikan uji, bukan akurasi identitas seluruh pertandingan. Pengujian regresi tambahan mencakup kandidat confidence rendah, jitter pergelangan kaki, kepala berbeda dengan pinggul bertumpuk, deduplikasi pose, dan pemulihan bertahap setelah kehilangan panjang. Laporan/klip hasil berada di `artifacts/doubles-improved-v2/` dan `artifacts/doubles-continuity-improved/`.

## Struktur

`badminton_vision/domain.py` memuat data dan geometri, `inference.py` model, `tracker.py` identitas roster, `worker.py` pemrosesan video di thread terpisah, `storage.py` log/ekspor, `render.py` overlay, dan `ui.py` antarmuka desktop. Keypoint dan timestamp disimpan untuk pengembangan klasifikasi pukulan berikutnya.

Referensi: [YOLO11](https://docs.ultralytics.com/models/yolo11/), [OSNet](https://arxiv.org/abs/1905.00953), [Qt QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).
#   b a d m i n t o n - a n a l y s i s  
 