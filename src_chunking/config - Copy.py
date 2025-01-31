import os
from src.custom_logging import Logger


# Folder Creations for setup
data_folder = 'data'
index_folder = os.path.join(data_folder, 'index')
llm_cache_folder = os.path.join(data_folder, 'llm_cache')
scraped_data_folder = os.path.join('tejas', 'tejas', 'files')
logs_folder = os.path.join(data_folder, 'logs')
reports_folder = os.path.join(data_folder, 'reports')
folders = [data_folder, index_folder, llm_cache_folder, scraped_data_folder, logs_folder, reports_folder]
for folder in folders:
    os.makedirs(folder, exist_ok=True)


# Caching
sentencize_cache_file = os.path.join(llm_cache_folder, 'sentencize_cache.json')
chat_openai_cache_file = os.path.join(llm_cache_folder, 'chat_openai_cache.pkl')
chat_gemini_cache_file = os.path.join(llm_cache_folder, 'chat_gemini_cache.pkl')


# Logging
log_file_name = 'logs.txt'
logger = Logger(os.path.join(logs_folder, log_file_name))


# Scraping Related
media_file_extensions = ['.jpg', '.png', 'jpeg', '.gif', '.mp4', '.pdf', '.doc', '.docx', '.xlsx', '.xlsm', '.xls', '.mp3', '.ashx', '.ppt', '.pptx', '.txt', '.zip', '.rar', '.7z', '.tar', '.gz', '.tgz', '.exe', '.msi', '.apk', '.ipa', '.dmg', '.pkg', '.deb', '.rpm', '.iso', '.img', '.bin', '.cue', '.mdf', '.nrg', '.vcd', '.avi', '.mkv', '.flv', '.mov', '.wmv', '.mpg', '.mpeg', '.m4v', '.webm', '.vob', '.3gp', '.3g2', '.m2v', '.m4v', '.f4v', '.f4p',]

# Embedding models
all_mpnet_base_v2 = 'all-mpnet-base-v2'
text_embedding_004 = 'text-embedding-004'
token_limits = {'all-mpnet-base-v2': 512, 'text-embedding-004': 2048}
default_embedding_model = 'text-embedding-004'


# LLM Related
gpt4o = 'gpt-4o'
gemini_15 = 'gemini-1.5-flash'
#openai_key = "insert key"
#gemini_key = 'insert key'


# Model usage customization
query_enhancement_model = gpt4o
query_enhancement_history_model = gpt4o


# Search parameters
twolevel_t1 = 0.6
twolevel_t2 = 0.8
twolevel_l1_reserve = 25 # percentage of k to reserve for l1 (large paras) index
twolevel_l2_reserve = 25 # percentage of k to reserve for l2 (small paras) index


# company names to index name
company_to_index_name = {
    'adanipower':'Adani Power',
    'tatapower':'Tata Power',
    'adckcl':'ADCKCL',
    'tejasnetworks':'Tejas Networks',
    'lauruslabs':'Laurus Labs',
    'drreddys':'Dr Reddys',
    'sanghvicranes':'Sanghvi Cranes',
    'zee': 'Zee Entertainment',
    'irb': 'IRB Infra',
    'larsentoubro': 'Larsen & Toubro',
    'orientindia': 'Orient Technologies'
}