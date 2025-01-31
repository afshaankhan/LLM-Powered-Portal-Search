import os
import json
import fitz
import spacy
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from src.config import sentencize_cache_file, logger, index_folder

def print_list(lst, title, element_delim=None):
    """
    Prints a given list of items with a title
    """
    logger.info(f'{title}:')
    for item in lst:
        logger.info(f'\t-->{item}')
        if element_delim is not None:
            logger.info(element_delim)
    logger.info('----------------------------------------')
    return

def load_pickle(filepath, default={}):
    if not os.path.exists(filepath):
        print(f'File not found: {filepath}')
        print('Hence returning default')
        return default
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_pickle(embedding_dict, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(embedding_dict, f)

def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    return

def write_text(text, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    return

def get_data_size(scrape_res, keys=None):
    """
    Given a dict, the function returns the total size of text in the dict.
    Args:
        scrape_res: dict
        keys: list of keys to consider for size calculation. If None, all keys are considered.
    Returns:
        total_size: int
    """
    total_size = 0
    if keys is None:
        keys = list(scrape_res.keys())
    for key in keys:
        total_size += len(scrape_res[key])
    return total_size

def sentencize(text):
    """
    The function sentencizes a given text.
    1. The text is split into chunks of 80000 characters (because thats the max limit for spacy)
    2. Each chunk is sentencized separately
    3. The sentences from all the chunks are concatenated into a single list
    Args:
        text: str
    Returns:
        sentences: list of strings
    """
    cache = load_json(sentencize_cache_file)
    if text in cache.keys():
        return cache[text]
    nlp = spacy.load("en_core_web_sm")
    text_chunks = [text[i:i+80000] for i in range(0, len(text), 80000)]
    sentences = []
    for chunk in text_chunks:
        doc = nlp(chunk)
        chunk_sents = [sent.text for sent in doc.sents]
        sentences.extend(chunk_sents)
    cache[text] = sentences
    write_json(cache, sentencize_cache_file)
    return sentences

def get_token_count(text):
    """
    compute token count for a given text
    """
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    return token_count

def indexfile_to_indexpath(indexname):
    company_folder = indexname.split('.')[0].split('_')[0]
    indexpath = os.path.join(index_folder, company_folder, indexname)
    return indexpath

def numpy_encoder(obj):
    return {
        '__ndarray__': True,
        'dtype': str(obj.dtype),
        'shape': obj.shape,
        'data': obj.tolist()
    }

def numpy_decoder(dct):
    return np.array(dct['data'], dtype=dct['dtype']).reshape(dct['shape'])

def read_pdf(pdf_path):
    pages = []
    doc = fitz.open(pdf_path)
    for i in range(doc.page_count):
        page = doc[i]
        page_text = page.get_text()
        pages.append(page_text)
    doc.close()
    return pages