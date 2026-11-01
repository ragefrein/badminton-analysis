# Model lokal

Model sudah tersedia di folder ini. Aplikasi tidak mengunduh apa pun ketika dijalankan.

| File | Fungsi | SHA-256 |
| --- | --- | --- |
| yolo11s-pose.pt | Deteksi orang dan 17 keypoint COCO | 1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba |
| osnet_x0_25_msmt17.pt | Embedding penampilan untuk pencocokan identitas | 6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99 |

Asal dan atribusi: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[OSNet / Kaiyang Zhou](https://github.com/KaiyangZhou/deep-person-reid),
[BoxMOT](https://github.com/mikel-brostrom/boxmot).
Arsitektur OSNet dimuat dari paket BoxMOT yang terpasang; bobot dimuat secara ketat agar model yang tidak cocok tidak dipakai diam-diam.
Ultralytics dan BoxMOT menyertakan ketentuan lisensi AGPL-3.0; lihat lisensi masing-masing dependensi untuk distribusi.

Jika bobot hilang, jalankan `python scripts/download_models.py`. URL unduhan dan checksum terdapat di script tersebut.
