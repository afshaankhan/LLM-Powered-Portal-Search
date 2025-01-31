import os
import math
from src.llm import chat
from tqdm import tqdm
from src.utils import print_list, get_data_size, write_json, load_json
from src.prompts import filter_non_investment_websites, relevant_url_retrieval_prompt, clean_webpage_content
from src.config import logger

def get_common_prefix(s1, s2):
    """
    Given 2 strings, the function finds the longest common prefix between the strings
    """
    prefix = ''
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix

def get_prefixes(strings):
    """
    Given a list of strings, the function finds all the longest prefixes between all the pairs of strings.
    Then we filter out prefixes to ensure that we have only those prefixes that end with /.
    Args:
        strings: list of strings
    Returns:
        prefixes: list of prefixes (strings)
    """
    prefixes = []
    for i in range(len(strings)):
        for j in range(i+1, len(strings)):
            prefix = get_common_prefix(strings[i], strings[j])
            last_occurence = prefix.rfind('/')
            prefix = prefix[:last_occurence+1]
            if prefix in prefixes or not prefix.endswith('/'):
                continue
            prefixes.append(prefix)
    return prefixes

def remove_superstrings(strings):
    """
    Given a list of strings, we remove all the superstrings. If 'ab' is a string, then 'abc' is a superstring.
    """
    strings = sorted(strings)
    for i in range(len(strings)-1, -1, -1):
        is_superstring = any([strings[j] in strings[i] for j in range(i)])
        if is_superstring:
            del strings[i]
    return strings

def get_unwanted_url_prefixes(prefixes):
    """
    Given a scraped result, the function finds the unwanted URL prefixes.
    The function uses llm for this.
    """
    prefixes = sorted(prefixes) # assuming it helps llm understand better
    prompt = filter_non_investment_websites.format('\n'.join(prefixes))
    raw_res, json_res = chat('gpt-4o', prompt, response_format='json')
    unwanted_urls = json_res['websites']
    unwanted_urls = remove_superstrings(unwanted_urls) # removing superstrings
    return unwanted_urls

def get_relevant_urls(urls, max_chunk_size=50):
    """
    Given a list of urls, the function chunks the urls and then uses llm to find the relevant urls.
    Args:
        urls: list of urls
        max_chunk_size: int: The maximum chunk size for the urls
    Returns:
        relevant_urls: list of relevant urls
    """
    # chunking urls
    num_chunks = math.ceil(len(urls)/max_chunk_size)
    chunk_size = len(urls)//num_chunks
    url_chunks = [urls[i:i+chunk_size] for i in range(0, len(urls), chunk_size)]

    relevant_urls = []
    for i in range(len(url_chunks)):
        prompt = relevant_url_retrieval_prompt.format('\n'.join(url_chunks[i]))
        raw_res, json_res = chat('gpt-4o', prompt, response_format='json')
        relevant_urls.extend(json_res['useful_websites'])
    return relevant_urls

def add_cleaned_text(scrape_res_path, cleaned_json_path = None):
    """
    Given a scrape result, the function cleans the text using llm and then saves the cleaned text in a json file in the same folder as the original file. The function uses gemini-1.5-flash for this.
    Args:
        scrape_res_path: str: The path to the scrape result
        cleaned_json_path: str: The path to save the cleaned json file
    Returns:
        scrape_res: dict: The cleaned scrape result
    """
    scrape_res = load_json(scrape_res_path)
    for key in tqdm(scrape_res.keys()):
        text = scrape_res[key]
        prompt = clean_webpage_content.format(text)
        raw_res, json_res = chat('gemini-1.5-flash', prompt, response_format='json')
        if json_res is None:
            print('json res is None')
            print(f'{raw_res.text=}')
        cleaned_text = json_res['cleaned_text']
        cleaned_text = cleaned_text.replace('\n', ' ')
        cleaned_text = cleaned_text.replace('*', ' ')
        cleaned_text = ' '.join(cleaned_text.split())
        scrape_res[key] = cleaned_text
    if cleaned_json_path is None:
        cleaned_json_path = scrape_res_path.replace('.json', '_cleaned.json')
    write_json(scrape_res, cleaned_json_path)
    return scrape_res

def remove_duplicates(scrape_res):
    std_scrape_res = {}
    for key in scrape_res.keys():
        value = scrape_res[key]
        key = key if key.endswith('/') else key+'/'
        std_scrape_res[key] = value
    return std_scrape_res

def clean_scrape_res(scrape_res):
    """
    Given a scrape result, the function cleans the scrape result by removing unwanted URLs and then finding the relevant URLs.
    Unwanted URLs are those that are not useful for making investment decisions.
    Relevant URLs are those that are useful for making investment decisions.
    (check the prompt for more details)
    Args:
        scrape_res: dict: The scrape result
    Returns:
        cleaned_scrape_res: dict: The cleaned scrape result
    """
    dedup_scrape_res = remove_duplicates(scrape_res)
    url_prefixes = get_prefixes(list(dedup_scrape_res.keys()))
    logger.info(f'No of prefixes: {len(url_prefixes)}')
    unwanted_urls_prefixes = get_unwanted_url_prefixes(url_prefixes)
    wanted_urls_prefixes = sorted(list(set(url_prefixes)-set(unwanted_urls_prefixes)))
    cleaned_scrape_res = {key:dedup_scrape_res[key] for key in dedup_scrape_res.keys() if not any([prefix in key for prefix in unwanted_urls_prefixes])}
    cur_urls = list(cleaned_scrape_res.keys())
    relevant_urls = get_relevant_urls(cur_urls)
    cleaned_scrape_res = {key:cleaned_scrape_res[key] for key in cleaned_scrape_res.keys() if key in relevant_urls}
    cleaned_scrape_res = {key:cleaned_scrape_res[key] for key in cleaned_scrape_res.keys() if cleaned_scrape_res[key].strip()}
    keys = list(cleaned_scrape_res.keys())
    print_list(unwanted_urls_prefixes, 'Unwanted URL prefixes')
    print_list(wanted_urls_prefixes, 'Wanted URL Prefixes')
    print(f'No of URLs:')
    print(f'\tTotal: {len(scrape_res)}')
    print(f'\tRelevant: {len(cleaned_scrape_res)}')
    print(f'Size of text:')
    print(f'\tTotal: {get_data_size(scrape_res)}')
    print(f'\tRelevant: {get_data_size(cleaned_scrape_res)}')
    print_list(relevant_urls, 'Relevant URLs')
    print_list(list(set(cur_urls)-set(relevant_urls)), 'Irrelevant URLs')
    print('----------------------------------------')
    return cleaned_scrape_res