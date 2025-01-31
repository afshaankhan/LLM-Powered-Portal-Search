import math
import numpy as np
from src.llm import chat
from src.embeddings import get_embeddings
from src.prompts import query_enhancement_prompt, query_enhancement_with_history
from src.config import logger, query_enhancement_history_model, query_enhancement_model, twolevel_t1, twolevel_t2, twolevel_l2_reserve, twolevel_l1_reserve

def query_enhancement(query, history=None, history_len=10):
    """
    Given a query, the function enhances the query by generating related phrases, sentences, questions.
    Args:
        query: str: The query to enhance
        history: list of dicts: The history of the conversation. Each dict has keys: 'role', 'content'
        history_len: int: The number of previous messages to consider for query enhancement
    """
    if history is None:
        prompt = query_enhancement_prompt.format(query)
        raw_res, json_res = chat(query_enhancement_history_model, prompt, response_format='json')
    else:
        history_len = min(history_len, len(history))
        history_str = []
        for i in range(1, history_len+1):
            prefix = 'Response: ' if history[-i]['role']=='assistant' else 'Question: '
            history_str = [prefix + history[-i]['content']] + history_str
        history_str = '\n'.join(history_str)
        prompt = query_enhancement_with_history.format(history_str, query)
        raw_res, json_res = chat(query_enhancement_model, prompt, response_format='json')
    return json_res['query']

def search_on_dict_index(query, index, embedding_model, history=None, k=20, search_query='hybrid'):
    """
    Given a query and an index, the function searches for the query in the index.
    Args:
        query: str: The query to search for
        index: dict: The index to search on
        embedding_model: str: The model to use for getting embeddings
        history: list of dicts: The history of the conversation. Each dict has keys: 'role', 'content'
        k: int: The number of results to return
        search_query: str: The type of query to search with: 'hybrid', 'enhanced', 'raw'
    """
    def get_similarity_scores(query, index):
        params = {'task_type': 'retrieval_query'} # will be used only if embedding_model is 'text-embedding-004'
        query_emb = get_embeddings([query], params=params, model=embedding_model)[0]
        index_embs = [index[key]['emb'] for key in range(len(index))]
        emb_matrix = np.array(index_embs) # shpe: (n*d); n - number of paragraphs, d - dimension of embeddings
        similarity_scores = emb_matrix @ query_emb # shape: (n)
        similarity_scores = similarity_scores.flatten()
        return similarity_scores
    
    def aggregate_similarity_scores(scores_1, scores_2):
        aggregates_scores = np.maximum(scores_1, scores_2)
        return aggregates_scores

    if search_query == 'hybrid':
        enhanced_query = query_enhancement(query, history)
        raw_query = query
    elif search_query == 'enhanced':
        enhanced_query = query_enhancement(query, history)
        raw_query = None
    elif search_query == 'raw':
        enhanced_query = None
        raw_query = query
    else:
        print('WARNING: Invalid search_query parameter. Using both enhanced and raw query for semantic search...')
        search_query = 'hybrid'
        enhanced_query = query_enhancement(query, history)
        raw_query = query

    logger.info('Getting embeddings for the query...')
    if search_query == 'raw':
        similarity_scores = get_similarity_scores(raw_query, index)
    elif search_query == 'enhanced':
        similarity_scores = get_similarity_scores(enhanced_query, index)
    else:
        similarity_scores_raw = get_similarity_scores(raw_query, index)
        similarity_scores_enhanced = get_similarity_scores(enhanced_query, index)
        similarity_scores = aggregate_similarity_scores(similarity_scores_raw, similarity_scores_enhanced)
        
    top_k_inds = np.argsort(similarity_scores)[-k:][::-1]
    ret_text = '\n'.join([index[i]['text'] for i in top_k_inds])
    search_res = {}
    search_res['result_text'] = ret_text
    search_res['top_k_indices'] = top_k_inds.tolist()
    search_res['top_k_similarity_scores'] = similarity_scores[top_k_inds].tolist()
    search_res['top_k_chunks'] = [index[i]['text'] for i in top_k_inds]
    search_res['enhanced_query'] = enhanced_query
    search_res['top_k_websites'] = [index[i]['info']['website'] for i in top_k_inds] if 'website' in index[0]['info'] else None
    return search_res

def search_on_twolevel_index(query, index, params, history=None, k=20, search_query='hybrid'):
    def get_search_res():
        l1_index = index['l1_index']
        l2_index = index['l2_index']
        l1_search_res = search_on_dict_index(query, l1_index, params['embedding_model'], history=history, k=len(l1_index), search_query=search_query)
        l2_search_res = search_on_dict_index(query, l2_index, params['embedding_model'], history=history, k=len(l2_index), search_query=search_query)
        return l1_search_res, l2_search_res
    
    def add_similarity_scores(l1_index, l2_index, l1_search_res, l2_search_res):
        for idx, sim_score in zip(l1_search_res['top_k_indices'], l1_search_res['top_k_similarity_scores']):
            l1_index[idx]['similarity_score'] = sim_score
        for idx, sim_score in zip(l2_search_res['top_k_indices'], l2_search_res['top_k_similarity_scores']):
            l2_index[idx]['similarity_score'] = sim_score
        return l1_index, l2_index
    
    def get_reserve_idx(l1_index, l2_index, params, l1_search_res, l2_search_res):
        l1_reserve = params.get('l1_reserve', twolevel_l1_reserve)
        l2_reserve = params.get('l2_reserve', twolevel_l2_reserve)
        num_l1_reserve = math.ceil(l1_reserve*k/100)
        num_l2_reserve = math.ceil(l2_reserve*k/100)
        l1_reserve_idx = l1_search_res['top_k_indices'][:num_l1_reserve]
        l1_reserve_cids = [cid for idx in l1_reserve_idx for cid in l1_index[idx]['cids']]
        l2_reserve_idx = [idx for idx in l2_search_res['top_k_indices'] if idx not in l1_reserve_cids][:num_l2_reserve] # making sure that no l2_reserve_idx is a child of l1_reserve_idx
        l1_reserve_idx = [{'type':'l1', 'idx': idx, 'similarity_score':l1_index[idx]['similarity_score'], 'category': 'l1_reserve'} for idx in l1_reserve_idx]
        l2_reserve_idx = [{'type':'l2', 'idx': idx, 'similarity_score':l2_index[idx]['similarity_score'], 'category': 'l2_reserve'} for idx in l2_reserve_idx]
        return l1_reserve_idx, l2_reserve_idx
    
    def get_aggregate_scores(l1_index, l2_index):
        aggregrates = []
        for pkey in l1_index.keys():
            if pkey in l1_reserve_idx:
                continue
            ckeys = l1_index[pkey]['cids']
            psim = l1_index[pkey]['similarity_score']
            for ckey in ckeys:
                csim = l2_index[ckey]['similarity_score']
                aggregrates.append([pkey, ckey, psim, csim, psim+csim/2])
        return aggregrates
    
    def get_remaining_top(aggregates, params, l1_reserve_idx, l2_reserve_idx):
        remaining_k = k - len(l1_reserve_idx) - len(l2_reserve_idx)
        t1 = params.get('t1', twolevel_t1)
        t2 = params.get('t2', twolevel_t2)
        prev_l1_idx = [entry['idx'] for entry in l1_reserve_idx]
        prev_l2_idx = [entry['idx'] for entry in l2_reserve_idx]
        aggregates = [agg for agg in aggregates if agg[0] not in prev_l1_idx and agg[1] not in prev_l2_idx]
        aggregates = sorted(aggregates, key=lambda x: (x[4], x[2], x[3]), reverse=True)
        remaining_idx = []
        cnt = 0
        for i in range(len(aggregates)):
            prev_l1_idx = [entry['idx'] for entry in remaining_idx if entry['type']=='l1'] + [entry['idx'] for entry in remaining_idx if entry['type']=='l1_l2']
            pkey, ckey, psim, csim, sim = aggregates[i]
            if pkey in prev_l1_idx:
                continue
            if psim >= t1 and csim >= t2:
                remaining_idx.append({'type':'l1_l2', 'idx': pkey, 'similarity_score': sim, 'category': 'l1_l2_rem', 'cidx': ckey})
            elif psim >= t1:
                remaining_idx.append({'type':'l1', 'idx': pkey, 'similarity_score': psim, 'category': 'l1_rem'})
            elif csim >= t2:
                remaining_idx.append({'type':'l2', 'idx': ckey, 'similarity_score': csim, 'category': 'l2_rem'})
            else:
                # break
                remaining_idx.append({'type':'l1_l2', 'idx': pkey, 'similarity_score': sim, 'category': 'l1_l2_bad', 'cidx': ckey})
            cnt += 1
            if cnt == remaining_k:
                break
        return remaining_idx

    def get_final_search_res(l1_index, l2_index, l1_reserve_idx, l2_reserve_idx, remaining_k_idx):
        search_res = {}
        result_text = ''
        top_k_indices = l1_reserve_idx + l2_reserve_idx + remaining_k_idx
        top_k_indices = sorted(top_k_indices, key=lambda x: x['similarity_score'], reverse=True)
        logger.info('------------------------------------')
        top_k_chunks = []
        for i in range(len(top_k_indices)):
            idx = top_k_indices[i]
            if idx['type'] == 'l1':
                pkey = idx['idx']
                text = l1_index[pkey]['text'] + '\n'
            elif idx['type'] == 'l2':
                ckey = idx['idx']
                text = l2_index[ckey]['text'] + '\n'
            else:
                pkey, ckey = idx['idx'], idx['cidx']
                text = l1_index[pkey]['text'] + '\n'
            result_text += text
            top_k_chunks.append(text)
            logger.info(f'{idx=}')
            logger.info(text)
            logger.info('------------------------------------')
        top_k_similarity_scores = [idx['similarity_score'] for idx in top_k_indices]
        top_k_indices_res = [{'category': idx['category'], 'idx': idx['idx']} for idx in top_k_indices]
        top_k_info = [l1_index[idx['idx']]['info'] if idx['type']=='l1' else l2_index[idx['idx']]['info'] for idx in top_k_indices]
        search_res['result_text'] = result_text
        search_res['top_k_indices'] = top_k_indices_res
        search_res['top_k_similarity_scores'] = top_k_similarity_scores
        search_res['top_k_chunks'] = top_k_chunks
        search_res['top_k_info'] = top_k_info
        return search_res

    l1_search_res, l2_search_res = get_search_res()
    l1_index, l2_index = add_similarity_scores(index['l1_index'], index['l2_index'], l1_search_res, l2_search_res)
    l1_reserve_idx, l2_reserve_idx = get_reserve_idx(l1_index, l2_index, params, l1_search_res, l2_search_res)
    aggregrates = get_aggregate_scores(l1_index, l2_index)
    remaining_k_idx = get_remaining_top(aggregrates, params, l1_reserve_idx, l2_reserve_idx)
    final_search_res = get_final_search_res(l1_index, l2_index, l1_reserve_idx, l2_reserve_idx, remaining_k_idx)
    final_search_res['enhanced_query'] = l1_search_res['enhanced_query']
    return final_search_res

def search_on_index(query, index, search_params, verbose=True):
    """
    The function performs search with a given query on a given index.
    Args:
        query: str: The query to search for
        index: The index to search on
        params: dict: Applicable only for two-level index:
            search_query: str: The type of query to search with: 'hybrid', 'enhanced', 'raw'
            history: if not None, the history will be used for query enhancement. The history is a list of dicts with system, user and assistant messages.
            k: int: The number of results to return
            t1: float: The threshold for similarity score at level 1
            t2: float: The threshold for similarity score at level 2
            l1_reserve: int: The percentage of k to reserve for level 1
            l2_reserve: int: The percentage of k to reserve for level 2
        output_path: str: The path to save the search output
        verbose: bool: Whether to print the search results
    Returns:
        search_res: dict: The search results
            keys:
                result_text: str: The text of the search results
                top_k_indices: list of int: The indices of the top k results
                top_k_similarity_scores: list of float: The similarity scores of the top k results
                top_k_chunks: list of str: The text of the top k results
                enhanced_query: str: The enhanced query
    """
    history = search_params.get('history', None)
    search_query = search_params.get('search_query', 'hybrid')
    k = search_params.get('k', 20)
    logger.info('Searching on two-level index...')
    search_params['embedding_model'] = index['embedding_model']
    search_res = search_on_twolevel_index(query, index, search_params, history=history, search_query=search_query, k=k)
    
    if verbose:
        logger.info('----------------------------------------------------')
        logger.info(f'Top {k} similarity scores: {search_res["top_k_similarity_scores"]}')
        logger.info(f'Top {k} indices: {search_res["top_k_indices"]}')
        logger.info(f'Enhanced query: {search_res["enhanced_query"]}')
        logger.info('----------------------------------------------------')

    return search_res