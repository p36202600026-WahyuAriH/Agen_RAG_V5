from pathlib import Path
import hashlib, json, os, re, time
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from get_embedding_function import get_embedding_function
from pdf_engine import extract_pdf, analyze_article, index_pdfs, FIELD_LABELS, export_matrix

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; CHROMA=ROOT/'chroma'; MATRIX=ROOT/'research_gap_matrix.xlsx'
DATA.mkdir(exist_ok=True)
st.set_page_config(page_title='Agen RAG ResearchGap Pro',page_icon='📚',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
:root{--ink:#172033;--muted:#667085;--line:#e5eaf2;--accent:#4f46e5}
.block-container{padding-top:1.5rem;padding-bottom:3rem;max-width:1500px}
.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(120deg,#171c3b,#4338ca 65%,#0891b2);color:white;margin-bottom:1rem}
.hero h1{font-size:2rem;margin:0 0 .35rem 0}.hero p{opacity:.88;margin:0}
div[data-testid="stMetric"]{border:1px solid var(--line);padding:14px 16px;border-radius:14px;background:white}
.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{padding:10px 16px;border-radius:10px}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px}
.small-note{color:#667085;font-size:.9rem}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📚 Agen RAG ResearchGap Pro</h1><p>Ekstraksi PDF yang lebih kuat · pencarian berbasis bukti · analisis artikel terstruktur · matriks gap penelitian.</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown('## ⚙️ Konfigurasi AI')
    model=st.text_input('Model Ollama',os.getenv('OLLAMA_MODEL','mistral'))
    embedding_model=st.text_input('Model embedding',os.getenv('EMBEDDING_MODEL','nomic-embed-text'),disabled=True)
    top_k=st.slider('Potongan konteks retrieval',3,15,7)
    chunk_size=st.slider('Ukuran potongan indeks',600,1800,1100,100)
    overlap=st.slider('Overlap potongan',50,300,180,10)
    st.caption('Ollama harus aktif. Model embedding harus sama saat indexing dan pencarian.')

t1,t2,t3,t4,t5=st.tabs(['🏠 Ringkasan','📄 Pustaka PDF','🧠 Tanya RAG','🔬 Ekstraksi Detail Artikel','📊 Research Gap & Evaluasi'])
pdfs=sorted(DATA.glob('*.pdf'))
with t1:
    c=st.columns(4)
    c[0].metric('Artikel PDF',len(pdfs)); c[1].metric('Indeks Chroma','Siap' if CHROMA.exists() and any(CHROMA.iterdir()) else 'Belum ada'); c[2].metric('Matriks','Tersedia' if MATRIX.exists() else 'Belum ada'); c[3].metric('Mode','Lokal / Ollama')
    st.markdown('### Alur kerja yang disarankan')
    st.markdown('1. Unggah PDF artikel pada **Pustaka PDF**.\n2. Bangun indeks untuk pencarian semantik.\n3. Gunakan **Tanya RAG** untuk pertanyaan lintas artikel dengan bukti halaman.\n4. Jalankan **Ekstraksi Detail Artikel** untuk membuat data terstruktur per artikel.\n5. Tinjau dan ekspor matriks; validasi DOI dan klaim ke PDF asli.')
    st.info('AI tidak boleh dianggap sebagai sumber primer. Semua temuan, DOI, dan research gap perlu diverifikasi manusia.')
with t2:
    uploads=st.file_uploader('Unggah satu atau beberapa artikel PDF',type=['pdf'],accept_multiple_files=True,key='upload')
    if st.button('💾 Simpan PDF',type='primary'):
        if not uploads: st.warning('Pilih PDF terlebih dahulu.')
        else:
            saved=0; duplicate=[]
            hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.pdf')}
            for f in uploads:
                content=f.getvalue(); digest=hashlib.sha256(content).hexdigest()
                if digest in hashes: duplicate.append(f.name); continue
                name=Path(f.name).name
                target=DATA/name
                if target.exists(): target=DATA/f'{Path(name).stem}_{digest[:8]}.pdf'
                target.write_bytes(content); hashes.add(digest); saved+=1
            st.success(f'{saved} PDF baru disimpan.')
            if duplicate: st.warning('Duplikat identik dilewati: '+', '.join(duplicate))
    if pdfs:
        st.dataframe(pd.DataFrame([{'Nama file':p.name,'Ukuran (KB)':round(p.stat().st_size/1024,1),'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for p in pdfs]),width='stretch',hide_index=True)
        a,b=st.columns(2)
        with a:
            if st.button('🔎 Bangun / perbarui indeks',type='primary'):
                with st.spinner('Mengekstrak teks dan membangun indeks…'):
                    try:
                        result=index_pdfs(DATA,CHROMA,chunk_size,overlap)
                        st.success(f"Indeks selesai. Halaman: {result['pages']} · potongan baru: {result['added']} · potongan terindeks: {result['total']}")
                    except Exception as e: st.error(f'Gagal membuat indeks: {e}')
        with b:
            if st.button('🧹 Reset indeks dan bangun ulang'):
                import shutil
                try:
                    if CHROMA.exists(): shutil.rmtree(CHROMA)
                    result=index_pdfs(DATA,CHROMA,chunk_size,overlap)
                    st.success(f"Indeks dibangun ulang: {result['total']} potongan.")
                except Exception as e: st.error(str(e))
    else: st.info('Belum ada PDF. Unggah artikel terlebih dahulu.')
with t3:
    question=st.text_area('Pertanyaan penelitian',placeholder='Contoh: Bandingkan metode, dataset, temuan, dan keterbatasan penelitian RAG untuk administrasi perguruan tinggi.',height=100)
    if st.button('🧠 Telusuri dan jawab',type='primary',key='ask'):
        try:
            if not CHROMA.exists() or not any(CHROMA.iterdir()): raise RuntimeError('Indeks belum tersedia. Bangun indeks pada tab Pustaka PDF.')
            if not question.strip(): raise ValueError('Pertanyaan tidak boleh kosong.')
            db=Chroma(persist_directory=str(CHROMA),embedding_function=get_embedding_function())
            results=db.similarity_search_with_score(question,k=top_k)
            if not results: st.warning('Tidak ditemukan konteks yang relevan.')
            else:
                context='\n\n'.join(f'[Sumber {i}] File: {d.metadata.get("source","?")} | halaman: {int(d.metadata.get("page",0))+1} | relevansi-distance: {score:.4f}\n{d.page_content}' for i,(d,score) in enumerate(results,1))
                prompt=f'''Anda adalah asisten peneliti akademik. Jawab dalam bahasa Indonesia. Gunakan hanya bukti dalam konteks. Bedakan fakta eksplisit, sintesis lintas artikel, dan interpretasi. Jangan mengarang DOI, angka, metode, atau sumber. Jika bukti kurang, katakan belum ditemukan. Setiap klaim penting harus menyebut [Sumber n] dan halaman. Jangan menyatakan bahwa hasil retrieval adalah tinjauan sistematis lengkap.\n\nKONTEKS:\n{context}\n\nPERTANYAAN: {question}\n\nJawaban berbasis bukti:'''
                with st.spinner('Menganalisis konteks dan menyusun jawaban…'):
                    answer=ChatOllama(model=model,temperature=0).invoke(prompt).content
                st.markdown('### Jawaban'); st.markdown(answer)
                st.markdown('### Bukti yang diambil')
                for i,(d,score) in enumerate(results,1):
                    with st.expander(f'[Sumber {i}] {d.metadata.get("source","?")} · halaman {int(d.metadata.get("page",0))+1} · distance {score:.4f}'):
                        st.write(d.page_content)
        except Exception as e: st.error(str(e))
with t4:
    st.write('Ekstrak metadata dan isi inti artikel ke kolom terstruktur. PDF scan dapat memerlukan OCR; hasil AI ditandai untuk verifikasi.')
    selected=st.multiselect('Pilih PDF untuk dianalisis',options=[p.name for p in pdfs],default=[p.name for p in pdfs[:5]])
    limit=st.slider('Batas karakter teks per artikel',10000,100000,50000,5000)
    if st.button('🧪 Ekstrak detail artikel',type='primary'):
        if not selected: st.warning('Pilih minimal satu artikel.')
        else:
            llm=ChatOllama(model=model,temperature=0,format='json'); rows=[]; bar=st.progress(0)
            for i,name in enumerate(selected):
                path=DATA/name
                with st.status(f'Menganalisis {name}',expanded=False) as status:
                    try:
                        extracted=extract_pdf(path,max_chars=limit,ocr=False)
                        if not extracted['text'].strip(): raise ValueError('Tidak ada teks yang dapat diekstrak. Kemungkinan PDF scan; jalankan OCR eksternal lalu coba lagi.')
                        data=analyze_article(llm,name,extracted)
                        data['source_file']=name; data['sha256']=hashlib.sha256(path.read_bytes()).hexdigest(); data['page_count']=extracted['page_count']; data['extraction_method']=extracted['method']; data['status']='Perlu verifikasi manusia'; data['error']=''
                        rows.append(data); status.update(label=f'Selesai: {name}',state='complete')
                    except Exception as e:
                        rows.append({'source_file':name,'status':'Gagal dianalisis','error':str(e)}); status.update(label=f'Gagal: {name}',state='error')
                bar.progress((i+1)/len(selected))
            new=pd.DataFrame(rows)
            out=ROOT/'article_details.xlsx'; export_matrix(new,out,sheet='Detail Artikel')
            st.session_state['last_details']=new
            st.success(f'Hasil tersimpan: {out.name}')
    if 'last_details' in st.session_state:
        details=st.session_state['last_details'].fillna('')
        st.dataframe(details,width='stretch',hide_index=True)
        st.download_button('⬇️ Unduh detail artikel (Excel)',(ROOT/'article_details.xlsx').read_bytes(),file_name='article_details.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
with t5:
    if MATRIX.exists():
        df=pd.read_excel(MATRIX).fillna(''); st.metric('Baris matriks',len(df)); st.dataframe(df,width='stretch',height=420,hide_index=True)
        st.download_button('⬇️ Unduh matriks Research Gap',MATRIX.read_bytes(),file_name=MATRIX.name,mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else: st.info('Matriks belum dibuat. Jalankan ekstraksi detail artikel, lalu ekspor sebagai matriks.')
    st.markdown('### Pemeriksaan sistem')
    checks=[{'Komponen':'Folder data','Status':'OK' if DATA.exists() else 'Tidak ada','Detail':str(DATA)}, {'Komponen':'PDF','Status':'OK' if pdfs else 'Belum ada','Detail':f'{len(pdfs)} file'}, {'Komponen':'Indeks Chroma','Status':'OK' if CHROMA.exists() and any(CHROMA.iterdir()) else 'Belum ada','Detail':str(CHROMA)}, {'Komponen':'Matriks Excel','Status':'OK' if MATRIX.exists() else 'Belum ada','Detail':str(MATRIX)}]
    st.dataframe(pd.DataFrame(checks),width='stretch',hide_index=True)
    st.caption('Untuk evaluasi tesis, siapkan query uji dengan jawaban acuan dan ukur retrieval (mis. Recall@k/MRR), kelengkapan ekstraksi, serta akurasi yang dinilai manusia.')
st.divider(); st.caption('Agen RAG ResearchGap Pro · Versi 5.0 · lokal-first · periksa setiap klaim terhadap PDF sumber sebelum dikutip.')
