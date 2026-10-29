"""PDF extraction, article-field extraction, and Chroma indexing utilities."""
from pathlib import Path
import hashlib, json, re
import pandas as pd
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function

UNKNOWN='Tidak ditemukan secara eksplisit dalam teks yang berhasil diekstrak'
FIELD_LABELS={
 'title':'Judul Paper','authors_year':'Penulis dan Tahun','doi':'DOI','problem':'Problem / Masalah Penelitian',
 'method':'Metode Penelitian','dataset':'Dataset / Data / Sampel','findings':'Temuan Utama','limitations':'Keterbatasan',
 'future_work':'Future Work','research_gap_candidate':'Kandidat Research Gap','evidence':'Bukti dan Halaman'}
KEYS=list(FIELD_LABELS)

def extract_pdf(path,max_chars=50000):
    """Extract page-aware text with pypdf; preserves page boundaries and metadata."""
    reader=PdfReader(str(path)); pages=[]; total=0
    for i,page in enumerate(reader.pages,1):
        try: text=(page.extract_text(extraction_mode='layout') or '').strip()
        except TypeError: text=(page.extract_text() or '').strip()
        if text:
            block=f'\n\n===== HALAMAN PDF {i} =====\n{text}'
            remaining=max_chars-total
            if remaining<=0: break
            pages.append(block[:remaining]); total+=min(len(block),remaining)
        if total>=max_chars: break
    alltext=''.join(pages)
    doi_matches=re.findall(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+',alltext,re.I)
    doi=doi_matches[0].rstrip('.,;)') if doi_matches else ''
    return {'text':alltext,'page_count':len(reader.pages),'pages_with_text':len(pages),'method':'pypdf-layout','doi_candidates':list(dict.fromkeys(doi_matches))[:10]}

def parse_json(raw):
    raw=raw.strip()
    try: return json.loads(raw)
    except json.JSONDecodeError:
        m=re.search(r'\{[\s\S]*\}',raw)
        if not m: raise ValueError('Model tidak menghasilkan JSON valid.')
        return json.loads(m.group(0))

def analyze_article(llm,filename,extracted):
    text=extracted['text']
    prompt=f'''Anda adalah analis literatur ilmiah yang sangat hati-hati. Ekstrak informasi artikel berdasarkan TEKS PDF di bawah. Keluarkan hanya JSON valid, semua nilai string. Kunci wajib: {', '.join(KEYS)}.
ATURAN KETAT:
1. Jangan mengarang. Bila tidak ada bukti, gunakan persis: "{UNKNOWN}".
2. DOI hanya dari teks; jangan menebak dari judul. Kandidat DOI terdeteksi secara regex: {extracted['doi_candidates']} — cocokkan dengan konteks, jangan otomatis percaya.
3. Untuk metode, jelaskan desain, algoritme/model, prosedur, dan evaluasi bila tersedia.
4. Dataset mencakup sumber, ukuran sampel, periode, fitur, pembagian train/test bila disebutkan.
5. Temuan sertakan angka dan kondisi eksperimen hanya bila eksplisit.
6. Keterbatasan hanya yang dinyatakan penulis; bila gap yang disimpulkan AI, beri label sebagai interpretasi.
7. Future work harus berasal dari bagian future work/conclusion jika ada.
8. Kolom evidence berisi kutipan pendek/parafrasa dan nomor halaman PDF. Setiap field penting perlu bukti halaman; jangan membuat nomor halaman.
9. Kandidat research gap adalah hipotesis, jelaskan dasar bukti dan apa yang belum diuji; jangan mengklaim gap pasti.
10. Judul paper jangan diambil dari nama file jika judul tidak ditemukan dalam teks; beri catatan bila memakai nama file sebagai kandidat.
NAMA FILE: {filename}
JUMLAH HALAMAN PDF: {extracted['page_count']}
TEKS ARTIKEL:
{text}
JSON:'''
    result=parse_json(llm.invoke(prompt).content)
    cleaned={k:str(result.get(k,UNKNOWN) or UNKNOWN).strip() for k in KEYS}
    # Enforce DOI evidence: only accept DOI patterns found in the extracted text.
    doi=cleaned.get('doi','')
    match=re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+',doi,re.I)
    if match: cleaned['doi']=match.group(0).rstrip('.,;)')
    elif doi and doi!=UNKNOWN: cleaned['doi']='Perlu verifikasi: '+doi
    cleaned['doi_candidates_detected']='; '.join(extracted['doi_candidates'])
    return cleaned

def index_pdfs(data_path,chroma_path,chunk_size=1100,overlap=180):
    data_path=Path(data_path); chroma_path=Path(chroma_path); pdfs=sorted(data_path.glob('*.pdf'))
    if not pdfs: raise ValueError(f'Tidak ada PDF di {data_path}')
    docs=[]; pages=0
    for pdf in pdfs:
        loaded=PyPDFLoader(str(pdf)).load(); pages+=len(loaded)
        for d in loaded:
            d.metadata['source']=pdf.name
            d.metadata['page']=int(d.metadata.get('page',0))
            d.metadata['sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
        docs.extend(loaded)
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap,separators=['\n\n','\n','. ','; ',' ',''])
    chunks=splitter.split_documents(docs)
    for c in chunks:
        src=Path(c.metadata.get('source','unknown.pdf')).name; page=c.metadata.get('page',0)
        digest=hashlib.sha256((src+str(page)+c.page_content).encode()).hexdigest()[:16]
        c.metadata['id']=f'{src}:p{page}:{digest}'
    db=Chroma(persist_directory=str(chroma_path),embedding_function=get_embedding_function())
    existing=set(db.get(include=[])['ids'])
    new=[c for c in chunks if c.metadata['id'] not in existing]
    if new: db.add_documents(new,ids=[c.metadata['id'] for c in new])
    return {'pages':pages,'chunks':len(chunks),'added':len(new),'total':len(existing)+len(new)}

def export_matrix(df,path,sheet='Research Gap'):
    path=Path(path)
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        df.to_excel(writer,index=False,sheet_name=sheet[:31]); ws=writer.sheets[sheet[:31]]
        ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        from openpyxl.styles import Alignment,Font,PatternFill
        for cell in ws[1]: cell.font=Font(bold=True,color='FFFFFF'); cell.fill=PatternFill('solid',fgColor='312E81'); cell.alignment=Alignment(wrap_text=True,vertical='top')
        for cells in ws.columns:
            letter=cells[0].column_letter
            vals=[len(str(c.value or '')) for c in cells[:50]]
            ws.column_dimensions[letter].width=min(max(max(vals or [12])+2,14),45)
            for cell in cells[1:]: cell.alignment=Alignment(wrap_text=True,vertical='top')
    return path
