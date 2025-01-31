import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.chunking import chunk_text_within_token_limit
import google.generativeai as genai
from src.config import gemini_key, token_limits, default_embedding_model, logger
genai.configure(api_key=gemini_key)

def get_embeddings(texts, params={}, model=None):
    """
    Compute embeddings for a list of texts. This is the function that is called by every other function to get embeddings. No other function is creating embeddings directly.
    Args:
        texts: list of strings
        params: dict specifying params needed for gemini embedding model
        model: str name of the model
    Returns:
        embeddings: numpy array of embeddings
    """
    model = default_embedding_model if model is None else model
    if model == 'all-mpnet-base-v2':
        logger.info(f'Embedding model - {model}')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model = model.to(device)
        embeddings = model.encode(texts, device=device)
    elif model == 'text-embedding-004':
        logger.info(f'Embedding model - {model}')
        task_type = params.get('task_type', 'retrieval_document')
        title = params.get('title', 'Web scraped raw data of a company')
        if task_type == 'retrieval_document':
            embeddings = genai.embed_content(model=f'models/{model}', content = texts, task_type=task_type, title = title)
        else:
            embeddings = genai.embed_content(model=f'models/{model}', content = texts, task_type=task_type)
        embeddings = embeddings['embedding']
        embeddings = np.array(embeddings)
    else:
        raise ValueError('Invalid model')
    return embeddings

def get_embeddings_wrapper(texts, params={}, model=None, batch_size=50):
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    print(f'{len(texts)=}')
    print(f'{len(batches)=}')
    all_embeddings = []
    for batch in tqdm(batches, file= sys.stdout):
        while True:
            time.sleep(1)
            sys.stdout.flush()
            try:
                embeddings = get_embeddings(batch, params=params, model=model)
                break
            except Exception as e:
                print(f'Error while embeddings creation: {e}')
                logger.info(f'Error while embeddings creation: {e}')
                time.sleep(60)
        all_embeddings.append(embeddings)
    embeddings = np.vstack(all_embeddings)
    return embeddings

def retrieve_similar_chunks(chunks, target, k = 2):
    # TODO: Add gemini embeddings
    """
    Computes top k similar chunks to the target chunk
    Args:
        chunks: list of strings
        target: string
        k: int
    Returns:
        top_k_inds: list of indices of top k similar chunks
    """
    sentences = [target] + chunks
    embeddings = get_embeddings(sentences)
    target_emb = embeddings[0].reshape(1, -1) # shape: (1*d); d - dimension of embeddings
    chunk_embs = embeddings[1:] # shape: (n*d); n - number of chunks, d - dimension of embeddings
    target_emb_norm = target_emb / np.linalg.norm(target_emb, axis=1, keepdims=True) # normalizing each column vector for computation of cosine sim
    chunk_embs_norm = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True) # normalizing each column vector for computation of cosine sim
    similarity_matrix = chunk_embs_norm @ target_emb_norm.T
    similarity_matrix = similarity_matrix.flatten()
    top_k_inds = np.argsort(similarity_matrix)[-k:][::-1]
    print(f'Similarity scores: {similarity_matrix}')
    print(f'{top_k_inds=}')
    print('top k similar chunks:')
    for i in top_k_inds:
        print(chunks[i])
    return top_k_inds.tolist()

def get_para_embedding(para, params={}, model=None):
    """
    Given a paragraph, the function computes the embedding for the paragraph as follows:
    1. The paragraph is chunked into smaller parts
    2. The embeddings for each chunk are computed
    3. The paragraph embedding is computed as the mean of the embeddings of the chunks
    """
    model = default_embedding_model if model is None else model
    token_limit = token_limits[model]
    chunks = chunk_text_within_token_limit(para, max_tokens=token_limit)
    embeddings = get_embeddings(chunks, params=params, model=model)
    para_embedding = np.mean(embeddings, axis=0)
    return para_embedding

def get_multipara_embedding(paras, params={}, model=None, batch_size=50):
    """
    Given a list of paras, the function computes the embeddings for all the paras.
    The function makes sure all the paras are passed to model at a time hence reducing the processing time.
    Args:
        paras: list: The list of paragraphs
    Returns:
        para_embeddings: list: The embeddings for each paragraph. Each embedding is a numpy array.
    """
    model = default_embedding_model if model is None else model
    token_limit = token_limits[model]
    all_chunks = [chunk_text_within_token_limit(para, max_tokens=token_limit) for para in paras]
    chunk_lens = [len(chunk) for chunk in all_chunks]
    all_texts = []
    for chunk in all_chunks:
        all_texts.extend(chunk)
    embeddings = get_embeddings_wrapper(all_texts, params=params, model=model, batch_size=batch_size)
    para_embeddings = []
    for i in range(len(chunk_lens)):
        prev_idx = sum(chunk_lens[:i])
        cur_idx = sum(chunk_lens[:i+1])
        para_embedding = np.mean(embeddings[prev_idx:cur_idx], axis=0)
        para_embeddings.append(para_embedding)
    return para_embeddings

def get_scrape_res_paras_embeddings(scrape_res_paras, params={}, model=None):
    """
    Given a dict of scraped paragraphs, the function computes the embeddings for each paragraph
    Args:
        scrape_res_paras: dict: The scraped paragraphs
    Returns:
        scrape_res_embeddings: dict: The embeddings for each paragraph. Here, each key is a website and the value is a list of embeddings for the paragraphs of the website. Each embedding is a numpy array.
    """
    scrape_res_embeddings = {}
    keys = list(scrape_res_paras.keys())
    para_counts = [len(scrape_res_paras[key]) for key in keys]
    all_paras = [para for key in keys for para in scrape_res_paras[key]]
    all_para_embeddings = get_multipara_embedding(all_paras, params=params, model=model)
    for i in range(len(keys)):
        prev_idx = sum(para_counts[:i])
        cur_idx = sum(para_counts[:i+1])
        para_embeddings = all_para_embeddings[prev_idx:cur_idx]
        scrape_res_embeddings[keys[i]] = para_embeddings
    return scrape_res_embeddings