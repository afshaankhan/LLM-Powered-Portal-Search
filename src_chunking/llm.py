import os
import json
import time
from openai import OpenAI
import google.generativeai as genai
from src.utils import load_pickle, write_pickle
from src.config import chat_openai_cache_file, chat_gemini_cache_file, openai_key, gemini_key, logger

client = OpenAI(api_key=openai_key)
genai.configure(api_key=gemini_key)

def chat_completion_openai(prompt, model='gpt-4o', messages=None, response_format=None, cache=True):
    """
    Given a prompt, this function returns the completion from the model.
    The function also caches the completions.
    Args:
        prompt: str: The prompt for the model
        model: str: The model to use for completion
        messages: list: The messages to use for completion
        response_format: str: The response format to use
        cache: bool: Whether to cache the completions
    Returns:
        res: str: The completion from the model
        pres: str: processed completion from the model
    """
    completions = load_pickle(chat_openai_cache_file)
    messages = messages if messages is not None else [{"role": "user", "content": prompt}]
    resp_format = response_format
    if response_format == 'json':
        response_format = {"type": "json_object"}
    if cache and prompt in completions:
        logger.info('using cache')
        res = completions[prompt]
    else:
        logger.info('not using cache')
        while True:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0,
                    stream=False
                )
                res = stream.choices[0].message.content
                if resp_format == 'json':
                    dict_res = json.loads(res)
                    res = res, dict_res
                else:
                    res = res, res
                break
            except json.decoder.JSONDecodeError:
                logger.info(f"{model} hasn't returned proper json during cleaning")
                logger.info('Retrying...')
            except Exception as e:
                logger.info('Unknown error openai:', e)
                logger.info('Sleeping for 5 seconds...')
                time.sleep(10)
    
    completions[prompt] = res
    write_pickle(completions, chat_openai_cache_file)
    return res

def chat_completion_gemini(prompt, messages=None, response_format=None, cache=True, max_json_retries = 3):
    completions = load_pickle(chat_gemini_cache_file)
    """
    Given a prompt, this function returns the completion from the model.
    The function also caches the completions.
    If model doesn't return proper json even after 3 retries, then the function uses gpt-4o model. It also caches the output of gpt-4o model.
    Args:
        prompt: str: The prompt for the model
        messages: list: The messages to use for completion
        response_format: str: The response format to use
        cache: bool: Whether to cache the completions
    Returns:
        res: str: The completion from the model
        pres: str: processed completion from the model
    """
    if response_format=='json':
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
    else:
        model = genai.GenerativeModel("gemini-1.5-flash")
    
    # cache lookup
    if cache and prompt in completions:
        res = completions[prompt]
    else:
        while True:
            try:
                res = model.generate_content(prompt if messages is None else messages)
                if response_format == 'json':
                    dict_res = json.loads(res.text)
                    res = res, dict_res
                else:
                    res = res, res.text
                break
            except json.decoder.JSONDecodeError:
                logger.info(f"gemini-1.5-flash hasn't returned proper json during cleaning")
                logger.info('Retrying...')
                max_json_retries -= 1
                if max_json_retries == 0:
                    logger.info('Max retries reached. Using gpt-4o...')
                    res = chat('gpt-4o', prompt, response_format='json')
                    break
            except Exception as e:
                logger.info('Unknown error gemini:', e)
                logger.info('Sleeping for 10 seconds...')
                time.sleep(10)
    
    completions[prompt] = res
    write_pickle(completions, chat_gemini_cache_file)
    return res

def chat(model, prompt, response_format = None, messages=None, cache=True):
    """
    Given a model and prompt, this function returns the completion from the model.
    Args:
        model: str: The model to use for completion
        prompt: str: The prompt for the model
        response_format: str: if 'json', then the response is returned as json object. Else, as string
        messages: list: The messages to use for completion
        cache: bool: Whether to cache the completions
    Returns:
        res: str: The completion from the model
        pres: str: processed completion from the model
    """
    if model == 'gpt-4o':
        res, pres = chat_completion_openai(prompt, model, messages, response_format, cache)
        return res, pres
    elif model == 'gemini-1.5-flash':
        res, pres = chat_completion_gemini(prompt, messages, response_format, cache)
        return res, pres
    
def start_covo(model):
    """
    Given a model, this function starts a conversation with the model.
    Args:
        model: str: The model to use for conversation
    Returns:
        messages: list: The messages from the conversation
    """
    print('Starting conversation...')
    messages = []
    if model == 'gpt-4o':
        while True:
            query = input('You: ')
            if query == 'exit':
                print('Exiting conversation...')
                break
            messages.append({"role": "user", "content": query})
            _, pres = chat(model, query, messages=messages, cache=False)
            print(f'Assistant: {pres}')
            messages.append({"role": "assistant", "content": pres})
        return messages
    elif model == 'gemini-1.5-flash':
        while True:
            query = input('You: ')
            if query == 'exit':
                print('Exiting conversation...')
                break
            messages.append({"role": "user", "parts": query})
            res, pres = chat(model, query, messages=messages, cache=False)
            print(f'Assistant: {pres}')
            messages.append(res.candidates[0].content)
        return messages

if __name__ == '__main__':
    # start_covo('gpt-4o')
    start_covo('gemini-1.5-flash')