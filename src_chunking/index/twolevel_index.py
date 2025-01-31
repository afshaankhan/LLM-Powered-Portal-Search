from tqdm import tqdm
from src.config import all_mpnet_base_v2, text_embedding_004, default_embedding_model
from src.chunking import chunk_scrape_res_to_paras, chunk_corpus_to_paras
from src.embeddings import get_scrape_res_paras_embeddings, get_multipara_embedding

def create_twolevel_index(name, data, l1_chunk_size=1500, l2_chunk_size=300, embedding_model=default_embedding_model):
    def get_l2_paras():
        l2_paras = {}
        for key in tqdm(keys):
            l1_paras_key = l1_paras[key]
            l2_paras_key = [chunk_corpus_to_paras(para, para_size=l2_chunk_size) for para in l1_paras_key]
            l2_paras[key] = l2_paras_key
        return l2_paras
    
    def flatten_l2_paras(l2_paras):
        l2_paras_flat = [para for key in keys for paras in l2_paras[key] for para in paras]
        rebuild_info = [len(keys), [len(l2_paras[key]) for key in keys], [[len(paras) for paras in l2_paras[key]] for key in keys]]
        return l2_paras_flat, rebuild_info
    
    def rebuild(flatlist, info):
        l2_embeddings = {}
        cnt = 0
        for i in range(info[0]):
            s1 = []
            for j in range(info[1][i]):
                s2 = []
                for k in range(info[2][i][j]):
                    s2.append(flatlist[cnt])
                    cnt += 1
                s1.append(s2)
            l2_embeddings[keys[i]] = s1
        return l2_embeddings

    keys = list(data.keys())
    print(f'Length of data: {sum([len(data[key]) for key in keys])}')

    print("Chunking to form L1 paras")
    l1_paras = chunk_scrape_res_to_paras(data, chunk_size=l1_chunk_size)

    print("Chunking to form L2 paras")
    l2_paras = get_l2_paras()

    print("Getting embeddings for L1 paras")
    params = {'task_type': 'retrieval_document', 'title': f'Scraped raw data of {name}'} if embedding_model == text_embedding_004 else {}
    l1_embeddings = get_scrape_res_paras_embeddings(l1_paras, model=embedding_model, params=params)

    print("Getting embeddings for L2 paras")
    l2_paras_flat, rebuild_info = flatten_l2_paras(l2_paras)
    if embedding_model == all_mpnet_base_v2:
        l2_embeddings_flat = get_multipara_embedding(l2_paras_flat, model=embedding_model)
    elif embedding_model == text_embedding_004:
        params = {'task_type': 'retrieval_document', 'title': f'Scraped raw data of {name}'}
        l2_embeddings_flat = get_multipara_embedding(l2_paras_flat, model=embedding_model, params=params)
    else:
        raise ValueError('Invalid embedding model')
    l2_embeddings = rebuild(l2_embeddings_flat, rebuild_info)

    print("Building L1 and L2 indexes")
    l1_index = {}
    l2_index = {}
    pid = 0
    cid = 0
    for i in range(len(keys)):
        key = keys[i]
        paras = l1_paras[key]
        embs = l1_embeddings[key]
        for j in range(len(paras)):
            l1_text = paras[j]
            l1_emb = embs[j]
            num_children = len(l2_paras[key][j])
            cids = list(range(cid, cid+num_children))
            l1_index[pid] = {'text': l1_text, 'emb': l1_emb, 'cids':cids, 'info':{'website':key}}
            for k in range(num_children):
                l2_text = l2_paras[key][j][k]
                l2_emb = l2_embeddings[key][j][k]
                l2_index[cid] = {'text': l2_text, 'emb': l2_emb, 'pid':pid, 'info':{'website':key}}
                cid += 1
            pid += 1
    
    index = {'embedding_model':embedding_model, 'name':name, 'type':'2level', 'l1_index':l1_index, 'l2_index':l2_index}
    return index

def create_twolevel_index_corpus(name, corpus, l1_chunk_size=1500, l2_chunk_size=300, embedding_model=default_embedding_model):
    print('Computing l1 chunks...')
    l1_chunks = chunk_corpus_to_paras(corpus, para_size=l1_chunk_size)

    print('Computing l2 chunks...')
    l2_chunks = [chunk_corpus_to_paras(chunk, para_size=l2_chunk_size) for chunk in tqdm(l1_chunks)]

    print('Computing embeddings for l1 chunks...')
    l1_embeddings = get_multipara_embedding(l1_chunks, model=embedding_model)

    print('Computing embeddings for l2 chunks...')
    l2_chunks_flat = [para for chunk in l2_chunks for para in chunk]
    l2_embeddings_flat = get_multipara_embedding(l2_chunks_flat, model=embedding_model)
    l2_chunk_lens = [len(chunk) for chunk in l2_chunks]
    l2_embeddings = []
    for i in range(len(l2_chunk_lens)):
        cur_chunk_embs = l2_embeddings_flat[:l2_chunk_lens[i]]
        l2_embeddings.append(cur_chunk_embs)
        l2_embeddings_flat = l2_embeddings_flat[l2_chunk_lens[i]:]
    
    print('Building l1 and l2 indexes...')
    l1_index = {}
    l2_index = {}
    pid = 0
    cid = 0
    for i in range(len(l1_chunks)):
        cids = list(range(cid, cid+l2_chunk_lens[i]))
        l1_index[pid] = {'text':l1_chunks[i], 'emb':l1_embeddings[i], 'cids':cids, 'info':{}}
        for j in range(len(l2_chunks[i])):
            l2_index[cid] = {'text':l2_chunks[i][j], 'emb':l2_embeddings[i][j], 'pid':pid, 'info':{}}
            cid += 1
        pid += 1
    
    index = {'embedding_model':embedding_model, 'name':name, 'type':'2level', 'l1_index':l1_index, 'l2_index':l2_index}
    return index