"""Batch-create a research-gap Excel matrix from local PDF articles using Ollama."""
import os, hashlib
from pathlib import Path
import pandas as pd
from langchain_ollama import ChatOllama
from pdf_engine import extract_pdf, analyze_article, FIELD_LABELS, export_matrix
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; MODEL=os.getenv('OLLAMA_MODEL','mistral')
def main():
    pdfs=sorted(DATA.glob('*.pdf'))
    if not pdfs: raise SystemExit(f'Tidak ada PDF di {DATA}')
    llm=ChatOllama(model=MODEL,temperature=0,format='json'); rows=[]; seen=set()
    for i,pdf in enumerate(pdfs,1):
        digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
        if digest in seen: print(f'Duplikat dilewati: {pdf.name}'); continue
        seen.add(digest); print(f'[{i}/{len(pdfs)}] {pdf.name}')
        try:
            ext=extract_pdf(pdf)
            if not ext['text'].strip(): raise ValueError('Tidak ada teks yang dapat diekstrak; kemungkinan PDF scan dan perlu OCR.')
            row=analyze_article(llm,pdf.name,ext); row.update(source_file=pdf.name,sha256=digest,page_count=ext['page_count'],extraction_method=ext['method'],status='Perlu verifikasi manusia',error='')
        except Exception as e:
            row={k:'' for k in list(FIELD_LABELS)+['doi_candidates_detected']}; row.update(source_file=pdf.name,sha256=digest,status='Gagal dianalisis',error=str(e)); print('  ERROR:',e)
        rows.append(row)
    df=pd.DataFrame(rows); out=ROOT/'research_gap_matrix.xlsx'; export_matrix(df,out)
    print(f'Matriks tersimpan: {out}')
if __name__=='__main__': main()
