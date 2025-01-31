import math
from tqdm import tqdm
from src.config import logger
from src.utils import get_token_count, sentencize

def chunk_text_within_token_limit(text, max_tokens = 512):
    """
    Chunk text into smaller parts
    """
    sents = sentencize(text)
    tokens_per_sent = [get_token_count(sent) for sent in sents]
    chunks = []
    cur_chunk = []
    cur_chunk_token_count = 0
    for i in range(len(sents)):
        sent = sents[i]
        num_tokens = tokens_per_sent[i]
        if cur_chunk_token_count + num_tokens <= max_tokens: # if adding the current sentence doesn't exceed the max_tokens
            cur_chunk.append(sent)
            cur_chunk_token_count += num_tokens
        elif num_tokens > max_tokens: # if the current sentence itself exceeds the max_tokens
            chunks.append(cur_chunk)
            chunks.append([sent])
            cur_chunk = []
            cur_chunk_token_count = 0
        else: # if adding the current sentence exceeds the max_tokens
            chunks.append(cur_chunk)
            cur_chunk = [sent]
            cur_chunk_token_count = num_tokens
    if len(cur_chunk) > 0:
        chunks.append(cur_chunk)
    chunk_texts = ['\n'.join(chunk) for chunk in chunks]
    chunk_texts = ['\n'.join(text.split('\n')) for text in chunk_texts]
    return chunk_texts

def chunk_corpus_to_paras(text, para_size = 800):
    """
    Given a large piece of text, the function chunks the text into paragraphs of size para_size.
    Args:
        text: str: The large piece of text
        para_size: int: The size of the paragraphs
    Returns:
        paras: list: The list of paragraphs
    """
    def can_partition(nums, max_sum, k):
        current_sum = 0
        parts = 1  # Start with 1 part
        
        for num in nums:
            current_sum += num
            if current_sum > max_sum:
                parts += 1
                current_sum = num  # Start new part
                if parts > k:
                    return False
        return True

    def find_k_partitions(nums, k):
        left, right = max(nums), sum(nums)  # Binary search range
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            # Check if we can partition into k or fewer parts with max_sum <= mid
            if can_partition(nums, mid, k):
                result = mid  # Possible answer
                right = mid - 1  # Try for a smaller maximum sum
            else:
                left = mid + 1  # Increase the maximum sum
        
        # Now we know the optimal max sum, let's find the actual partition points
        current_sum = 0
        cuts = []
        current_parts = 1
        
        for i in range(len(nums)):
            current_sum += nums[i]
            if current_sum > result:
                cuts.append(i)  # Make a cut here
                current_sum = nums[i]  # Start new part
                current_parts += 1
                if current_parts == k:  # Stop early, no more cuts needed
                    break
        return cuts

    def chunk_sent(sent, max_len):
        num_chunks = math.ceil(len(sent) / max_len)
        piece_len = math.ceil(len(sent) / num_chunks)
        sents = []
        for i in range(num_chunks):
            sents.append(sent[i*piece_len:(i+1)*piece_len])
        return sents

    def chunk_sents(sents):
        res_sents = []
        for sent in sents:
            res_sents.extend(chunk_sent(sent, para_size))
        return res_sents

    sents = sentencize(text)
    sents = chunk_sents(sents)
    num_paras = math.ceil(len(text) / para_size)
    if num_paras == 1:
        paras = ['\n'.join(sents)]
        return paras
    sents_lens = [len(sent) for sent in sents]
    
    cuts = find_k_partitions(sents_lens, num_paras)
    if len(cuts) == 0:
        return [text]
    paras = []
    paras.append(sents[:cuts[0]])
    for i in range(1, len(cuts)):
        paras.append(sents[cuts[i-1]:cuts[i]])
    paras.append(sents[cuts[-1]:])
    paras = ['\n'.join(para) for para in paras]

    # logger.info('========================================')
    # logger.info(f'{len(text)=}')
    # logger.info(f'{sents_lens=}')
    # logger.info(f'{para_size=}')
    # logger.info(f'{num_paras=}')
    # logger.info(f'{cuts=}')
    # logger.info(f'{[len(para) for para in paras]=}')
    # logger.info('========================================')
    return paras

def chunk_corpus_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    # num_chunks = math.ceil(len(text) / (chunk_size-overlap_size))
    num_chunks = round(len(text) / (chunk_size))
    effective_chunk_size = math.ceil(len(text)/num_chunks)
    while True:
        chunk = text[:effective_chunk_size+overlap_size]
        chunks.append(chunk)
        text = text[effective_chunk_size:]
        if len(text) == 0:
            break
    return chunks

def chunk_scrape_res_to_paras(scrape_res, chunk_size = 800):
    """
    Given a scraped result, the function chunks the scraped result into paragraphs.
    Args:
        scrape_res: dict: The scraped result
    Returns:
        scrape_res_paras: dict: The scraped result with paragraphs. Here each key is a website and the value is a list of paragraphs.
    """
    scrape_res_paras = {}
    for key in tqdm(scrape_res.keys()):
        scrape_res_paras[key] = chunk_corpus_to_paras(scrape_res[key], para_size=chunk_size)
    return scrape_res_paras