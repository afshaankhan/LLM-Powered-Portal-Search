scraping challenges:
- too much data -> 2 rounds of cleaning
- lot of old data -> s

Challenges:
    Scraping:
        - Old data
            * checking time stamps in html response
            * using llama to determine if a webpage content is old or not

    Scrape Res cleaning:
        - remove unwanted urls from prefixs (Done)
        - obtain latest and investment related urls from all remaining urls (Done)

    Index Creation:
        - Remove duplicate paras using similarity scores (may be > 0.99)

    Chunking web page text to paras:
        - Strict Enforcing of max para size

    Embedding optimization:
        - providing gpu compatibility
        - passing entire list of paras of scrape res instead of para by para
        
    Improving conversation on index:
        - Ignoring chunks under certain similarity score
        - adding previous response of model to prompt
        - prompt modification telling model to consider previous responses
        - Have only past 5 response in conversation
        - using google's embedding model
        - trying aqc type of search rather than using embedding model
            * may be use cluster level summaries?
        - comparision of index conversation with and without cleaning
        - using previous response for query enhancement
        - ensuring only company related statements or phrases in query enhancement

