import os
from src.llm import chat
from src.utils import write_json
from src.config import logs_folder, logger
from src.index.utils import get_index_name
from src.index.search import search_on_index
from src.prompts import index_questioning, index_questioning_system_prompt, index_questioning_two_companies

def start_convo_on_index(index, search_params, conversation_file = 'conversation.json'):
    """
    Given a query and an index, the function starts a conversation on the index.
    Args:
        index: multi-level index
        params: dict: Applicable only for two-level index:
            Common Keys:
                query_enhance: bool: If True, the query will be enhanced before searching
                query_enhance_with_history: if True, the query is enhanced using the history of the conversation
                k: int: The number of results to return
                t1: float: The threshold for similarity score at level 1
                t2: float: The threshold for similarity score at level 2
                l1_reserve: int: The percentage of k to reserve for level 1
                l2_reserve: int: The percentage of k to reserve for level 2
        conversation_file: str; name of the file to save the conversation
    """
    logger.info('Starting conversation on index,..')
    query_enhance_with_history = search_params.get('query_enhance_with_history', True)
    history = []
    convo_path = os.path.join(logs_folder, conversation_file)
    while True:
        query = input('You: ')
        if query == 'exit':
            break
        logger.print('-----------------------------------------------')
        search_params['history'] = history if query_enhance_with_history else None
        search_res = search_on_index(query, index, search_params)
        sim_text = search_res['result_text'].replace('*', '')
        enhanced_query = search_res['enhanced_query']
        prompt = index_questioning.format(query, sim_text)
        system_message = [{'role': 'system', 'content': index_questioning_system_prompt}]
        messages = system_message + [{'role':entry['role'], 'content':entry['content']} for entry in history]
        messages.append({'role': 'user', 'content': prompt})
        history.append({'role': 'user', 'content': query, 'enhanced_query':enhanced_query, 'similar_text': sim_text.split('\n')})

        res, pres = chat('gpt-4o', '', messages=messages, cache=False)

        logger.print('Assistant:', pres.replace('*', ''))
        logger.print('-----------------------------------------------')
        history.append({'role': 'assistant', 'content': pres})
        history_log = {i:history[i] for i in range(len(history))}
        write_json(history_log, convo_path)
    return

def start_convo_on_two_indexes(index_1, index_2, search_params, conversation_file = 'conversation.json'):
    """
    Given a query and an index, the function starts a conversation on the index.
    Args:
        index_1: multi-level index
        index_2: multi-level index
        params: dict: Applicable only for two-level index:
            Common Keys:
                query_enhance: bool: If True, the query will be enhanced before searching
                query_enhance_with_history: if True, the query is enhanced using the history of the conversation
                k: int: The number of results to return
                t1: float: The threshold for similarity score at level 1
                t2: float: The threshold for similarity score at level 2
                l1_reserve: int: The percentage of k to reserve for level 1
                l2_reserve: int: The percentage of k to reserve for level 2
        conversation_file: str; name of the file to save the conversation
    """
    logger.info('Starting conversation on index,..')
    query_enhance_with_history = search_params.get('query_enhance_with_history', True)
    history = []
    convo_path = os.path.join(logs_folder, conversation_file)
    while True:
        query = input('You: ')
        if query == 'exit':
            break
        logger.print('-----------------------------------------------')
        search_params['history'] = history if query_enhance_with_history else None
        k = search_params['k']
        search_params['k'] = k//2
        index_1_name = get_index_name(index_1)
        index_2_name = get_index_name(index_2)

        search_res_1 = search_on_index(query, index_1, search_params)
        sim_text_1 = search_res_1['result_text'].replace('*', '')
        enhanced_query = search_res_1['enhanced_query']

        search_res_2 = search_on_index(query, index_2, search_params)
        sim_text_2 = search_res_2['result_text'].replace('*', '')
        search_params['k'] = k # reset k

        index_1_context = f'<<<{index_1_name}>>>\n```{sim_text_1}```'
        index_2_context = f'<<<{index_2_name}>>>\n```{sim_text_2}```'
        prompt = index_questioning_two_companies.format(query, index_1_context, index_2_context)

        system_message = [{'role': 'system', 'content': index_questioning_system_prompt}]
        messages = system_message + [{'role':entry['role'], 'content':entry['content']} for entry in history]
        messages.append({'role': 'user', 'content': prompt})

        sim_text = index_1_context + '\n' + index_2_context
        history.append({'role': 'user', 'content': query, 'enhanced_query':enhanced_query, 'similar_text': sim_text.split('\n')})

        res, pres = chat('gpt-4o', '', messages=messages, cache=False)

        logger.print('Assistant:', pres.replace('*', ''))
        logger.print('-----------------------------------------------')
        history.append({'role': 'assistant', 'content': pres})
        history_log = {i:history[i] for i in range(len(history))}
        write_json(history_log, convo_path)
    return