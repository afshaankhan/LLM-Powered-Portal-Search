import os
from src.index.conversation import start_convo_on_index, start_convo_on_two_indexes
from src.utils import load_pickle, indexfile_to_indexpath
from src.config import twolevel_t1, twolevel_t2, twolevel_l1_reserve, twolevel_l2_reserve

# Index conversation
index_name = 'tatapower_cleaned_1500_300_EMtex_2level.pkl'
index_path = indexfile_to_indexpath(index_name)
index = load_pickle(index_path)
search_params = {
    'search_query': 'enhanced',
    'query_enhance_with_history': True,
    'k': 40,
    't1': twolevel_t1,
    't2': twolevel_t2,
    'l1_reserve': twolevel_l1_reserve,
    'l2_reserve': twolevel_l2_reserve,
    }
start_convo_on_index(index, search_params, conversation_file='convo.json')

# # Comparision conversation
# index_name_1 = 'adanipower_cleaned_1500_300_2level.pkl'
# index_name_2 = 'tatapower_1500_300_EMtex_2level.pkl'
# index_path_1 = indexfile_to_indexpath(index_name_1)
# index_path_2 = indexfile_to_indexpath(index_name_2)
# index_1 = load_pickle(index_path_1)
# index_2 = load_pickle(index_path_2)
# search_params = {
#     'search_query': 'enhanced',
#     'query_enhance_with_history': True,
#     'k': 40,
#     't1': twolevel_t1,
#     't2': twolevel_t2,
#     'l1_reserve': twolevel_l1_reserve,
#     'l2_reserve': twolevel_l2_reserve,
#     }
# start_convo_on_two_indexes(index_1, index_2, search_params, conversation_file='convo.json')