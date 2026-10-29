# Agen RAG ResearchGap Pro — v5.0

Aplikasi Streamlit lokal untuk membaca artikel PDF, mengindeks konten dengan ChromaDB, menjawab pertanyaan berbasis RAG, mengekstrak detail paper, dan mengekspor matriks research gap. LLM/embedding dijalankan melalui Ollama.

## Fitur baru v5
- Dashboard ringkas dengan status PDF, indeks, matriks, dan alur penggunaan.
- Upload batch PDF, hash SHA-256, pencegahan duplikat identik, dan pengamanan nama file.
- Ekstraksi PDF berbasis halaman memakai `pypdf` layout mode; deteksi kandidat DOI dengan regex.
- Ekstraksi terstruktur: Judul Paper, Penulis/Tahun, DOI, Problem, Metode, Dataset, Temuan, Keterbatasan, Future Work, Kandidat Research Gap, Bukti/Halaman.
- Prompt lebih ketat: tidak mengarang, membedakan fakta eksplisit dengan interpretasi, menandai informasi yang tidak ditemukan.
- RAG menampilkan potongan sumber, nomor halaman, dan distance retrieval.
- Ekspor Excel dengan filter, freeze header, wrap text, kolom lebar, dan status verifikasi.
- Pengindeksan incremental dan tombol reset indeks.
- Status sistem dan catatan evaluasi untuk eksperimen tesis.

## Instalasi Windows PowerShell
```powershell
cd E:\00 Python\Agen_RAG_V5
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Install Ollama dari situs resmi, lalu jalankan:
```powershell
ollama pull nomic-embed-text
ollama pull mistral
python -m streamlit run app.py
```
Buka `http://localhost:8501`.

## Cara pakai
1. Unggah artikel PDF pada tab **Pustaka PDF**.
2. Klik **Bangun/perbarui indeks** untuk menyiapkan RAG.
3. Tanyakan hal spesifik pada tab **Tanya RAG**. Jawaban harus diperiksa ke sumber yang tampil.
4. Pilih PDF pada **Ekstraksi Detail Artikel**, jalankan analisis, lalu unduh `article_details.xlsx`.
5. CLI batch: `python research_gap_matrix.py` menghasilkan `research_gap_matrix.xlsx`.

## Catatan PDF scan / OCR
Versi ini meningkatkan ekstraksi teks dan menjaga nomor halaman, tetapi `pypdf` tidak melakukan OCR pada halaman berupa gambar. Jika PDF hasil scan, jalankan OCR terlebih dahulu (misalnya OCRmyPDF/Tesseract), kemudian unggah PDF hasil OCR. Untuk pemrosesan OCR otomatis, pasang OCRmyPDF/Tesseract dan integrasikan sesuai lingkungan sistem.

## Variabel lingkungan opsional
- `OLLAMA_MODEL` — model chat, default `mistral`.
- `EMBEDDING_MODEL` — model embedding, default `nomic-embed-text`.

## Batasan ilmiah
- DOI yang terdeteksi adalah kandidat, bukan verifikasi metadata eksternal.
- RAG bergantung pada kualitas teks, chunking, embedding, dan cakupan dokumen yang diindeks.
- Kandidat research gap merupakan hipotesis berbasis korpus yang diunggah, bukan bukti bahwa gap belum pernah diteliti.
- Verifikasi manual terhadap artikel asli wajib dilakukan sebelum sitasi/tesis. Untuk evaluasi, siapkan ground truth dan ukur retrieval (Recall@k/MRR), ketepatan field, serta agreement penilai.
