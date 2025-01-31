import os
import time
from src.scrape_cleaning import clean_scrape_res
from src.utils import write_pickle, load_json, read_pdf
from src.index.twolevel_index import create_twolevel_index, create_twolevel_index_corpus
from src.config import index_folder, scraped_data_folder,reports_folder, default_embedding_model

def json_to_twolevel_index(json_path, index_name, l1_chunk_size=1500, l2_chunk_size=300, embedding_model=default_embedding_model, cleaning=False):
    scrape_res = load_json(json_path)
    if cleaning:
        scrape_res = clean_scrape_res(scrape_res)
    index = create_twolevel_index(index_name, scrape_res, l1_chunk_size=l1_chunk_size, l2_chunk_size=l2_chunk_size, embedding_model=embedding_model)
    return index

def corpus_to_twolevel_index(corpus, index_name, l1_chunk_size=1500, l2_chunk_size=300, embedding_model=default_embedding_model):
    index = create_twolevel_index_corpus(index_name, corpus, l1_chunk_size=l1_chunk_size, l2_chunk_size=l2_chunk_size, embedding_model=embedding_model)
    return index

def pdf_to_twolevel_index(pdfpath, index_name, l1_chunk_size=1500, l2_chunk_size=300, embedding_model=default_embedding_model):
    pages = read_pdf(pdfpath)
    corpus = '\n\n'.join(pages)
    print(f'Size of pdf corpus: {len(corpus)}')
    index = create_twolevel_index_corpus(index_name, corpus, l1_chunk_size=l1_chunk_size, l2_chunk_size=l2_chunk_size, embedding_model=embedding_model)
    return index

def config_to_index(index_config):
    t1 = time.time()
    filename = index_config['filename']
    index_name = index_config['index_name']
    embedding_model = index_config.get('embedding_model', default_embedding_model)
    cleaning = index_config.get('cleaning', True)

    l1_chunk_size = index_config['l1_chunk_size']
    l2_chunk_size = index_config['l2_chunk_size']
    if filename.endswith('.json'):
        indexfile = filename.replace('.json', f'_{l1_chunk_size}_{l2_chunk_size}_EM{embedding_model[:3]}.pkl')
        scraped_file_path = os.path.join(scraped_data_folder, filename)
        index = json_to_twolevel_index(scraped_file_path, index_name, l1_chunk_size=l1_chunk_size, l2_chunk_size=l2_chunk_size, embedding_model=embedding_model, cleaning=cleaning)
    elif filename.endswith('.pdf'):
        indexfile = filename.replace('.pdf', f'_{l1_chunk_size}_{l2_chunk_size}_EM{embedding_model[:3]}.pkl')
        report_path = os.path.join(reports_folder, filename)
        index = pdf_to_twolevel_index(report_path, index_name, l1_chunk_size=l1_chunk_size, l2_chunk_size=l2_chunk_size, embedding_model=embedding_model)
    
    
    company_folder = filename.split('.')[0].split('_')[0]
    if 'index_path' in index_config:
        index_path = index_config['index_path']
    else:
        if 'index_folder' in index_config:
            index_save_folder = index_config['index_folder']
        else:
            index_save_folder = os.path.join(index_folder, company_folder)
        index_path = os.path.join(index_save_folder, indexfile)
    print(f'Creating index at: {index_path}')
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    write_pickle(index, index_path)
    t2 = time.time()
    print(f'Index created in {t2-t1} seconds')
    return index
