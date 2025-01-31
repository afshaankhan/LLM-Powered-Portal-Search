clean_webpage_content = """
Give below is piece of text extracted from a website. Its a raw text. The text is enclosed within triple backticks. Since the text is scraped, it has lot of meaning less phrases or words. The idea behind cleaning is to make the text look like a human written text. I don't want to lose on information after cleaning.

raw text: ```{}```

Output format:
Produce the output in json format where the key is "cleaned_text" and value is the cleaned text.
"""

filter_non_investment_websites = """
I want to invest in a company. I have collected multiple websites that belong to the company. Given a list of websites enclosed in triple backticks, I want to know which of these websites are not useful for making investment decisions. Since investment is an important thing, I want to make sure that I am not missing any information. A website should be considered totally useless or irrelevant to investment if and if only if the website is about atleast one of the below aspects:
    1. pricing/ cart/ checkout/ payment/ refund
    2. careers/ jobs
    3. contact us/ support/ customer service/ faq/ help
    4. terms and conditions/ legal/ policy/ privacy
    5. cookies
    6. login/ sign up/ register
    7. social media/ share/ like/ follow

```
{}
```

Output format:
Return the output in json format where the key is websites and value is list of websites that are not useful for making investment decisions. If you are unsure about a website, do not include it in the output since it is bad to make wrong decisions in investment.
"""

relevant_url_retrieval_prompt = """
Given a list of website enclosed in triple backticks, your goal is select the websites that are useful for making investment decisions. Since investment is an important thing, you want to make sure that you are not missing any information.

Below are list of aspects which are useful for investment decision making:
    1. intro about company (usually available in homepage)
    2. products, services, applications of the company
    3. Mission and vision of the company
    4. Latest (year 2024 or 2023) news, annoucements and events
    5. Partners of the company
    6. Technologies, equipment, practices used by the company
    7. Management team of the company
    8. clients of the company
    9. Orders or contracts won by the company
    10. Future plans, strategies of the company

Below are list of aspects which are not useful for investment decision making:
    1. pricing/ cart/ checkout/ payment/ refund
    2. careers/ jobs
    3. contact us/ support/ customer service/ faq/ help
    4. terms and conditions/ legal/ policy/ privacy
    5. cookies
    6. login/ sign up/ register
    7. social media/ share/ like/ follow
    8. news, events, annoucements older than 2023

Websites:
```
{}
```

Output format:
Provide your output in json format where the key is "useful_websites" and value is list of websites that are useful for making investment decisions. If name of the website is not very descriptive, you can include it since we don't want to miss out on anything.
"""

query_enhancement_prompt = """
Given the short query below, enclosed in triple backticks, generate a list of related phrases, sentences, questions that capture the core ideas and context of the query. The query will be used to perform similarity searches on raw scraped data that has been chunked into smaller parts. The goal is to enrich the query with additional relevant concepts and context to improve the effectiveness of the similarity search results. So make sure the output is diverse and covers a wide range of related concepts. Provide atleast 10 related phrases, sentences or questions that can enhance the original query.

Query: ```{}```

Output format:
The output must provided as a json where key is "query" and value is string containing 10 or more phrases.
"""

query_enhancement_with_history = """
I will give you a short and most recent history of chat between me and a chatbot. I am using chatbot to find out some details about a company to get insights on investing in the company. Each question I asked to the chatbot, I have attached additional data so that the chatbot can lookup. However, I am not putting that additional data here. Along with the history, I will give a query which I want to ask the chatbot. Right now, I perform a similarity search between query and the data base I have to get relevant information from raw text corpus. But sometimes, the current query doesn't capture the entire context and hence the similarity search results are bad. So I want you to enhance the query by generating related phrases. The enhanced query is used only for similarity search but not passed to the chatbot I am operating with. The goal is to enrich the query with additional relevant concepts and context to improve the effectiveness of the similarity search results. So make sure the output is diverse and covers a wide range of related concepts. Provide atleast 10 related phrases that can enhance the original query. The history is provide starting from the oldest to the most recent chat. The queries start with "Question: " and the responses start with "Response: ". The current query is enclosed within triple backticks.

History:
{}

Current query: ```{}```

Output format:
The output must provided as a json where key is "query" and value is string containing 10 or more phrases. since the goal is to enhance similarity search, provide phrases that improve similarity search results. Since the goal is to enhance similarity to search, you do not need to produces sentences or questions. Just phrases are enough.

Example:
History:
Question: What are the technologies used by the company? Give one line explanation for each technology.
Response: Based on the provided data, here are more details about Tejas Networks' offerings:
1. Direct To Mobile Broadcast Solution: Enables direct communication and content delivery to mobile devices.
2. MPLS-TP (Multiprotocol Label Switching - Transport Profile): A technology for efficient and reliable data transport across networks.
3. Packet Transport Network: Provides high-performance data transport for various applications.
4. Utility Network Modernization: Helps modernize utility networks with advanced technologies.
5. Fixed Mobile Convergence: Integrates fixed and mobile networks for seamless communication.

Current query: ```Explain the above technologies with more details. Add 5 to 6 sentences for each technology.```

In this case the current query is asking how detailed explanations. You should enhance in such a way that you add more keywords and phrases from the above 5 technologies in the response. You need not mention anything related to addition of 5 to 6 sentences for each technology. Hence the output should look something like this:
Output:
{{
    "query": "comprehensive overview of the technologies utilized by Tejas Networks. benefits and features of each technology listed by Tejas Networks. Benefits of Fixed Mobile Convergence to consumers and service providers. direct communication and content delivery to mobile devices using direct to mobile broadcast solution? Efficient and reliable data transport using MPLS-TP (Multiprotocol Label Switching - Transport Profile? what is packet transport network? how does the company provide high performance data transport. List of advanced technologies used for utility network modernization. How does fixed mobile convergence integrate fixed and mobile networks for seamless communication?",
}}
In the above sample output, there is no mention of 5 to 6 sentences for each technology since that doesn't help with similarity search. Only the phrases that enhance the similarity search are included.
"""

index_questioning = """
Given a piece of text enclosed with triple backticks about a company. The data given to you is scraped data from the company website. I want you to look at the data and answer the below question.

Question: {}

data: ```{}```

Steps to follow:
1. Read the question carefully.
2. Look at the data given.
3. Classify the question into one of the below categories:
    a. Company specific question: The question is very specific to the company. You should answer based on the data given.
    b. General question: The question is general. You can answer based on your knowledge.
    c. Mix of both: The question is mix of both. You should answer based on the data given and also based on your knowledge.
4. Check if the company is making any false claims in the data. You can use your knowledge to verify the claims.
5. Answer the question as follows:
    a. Keep the answer consise and to the point.
    b. Use simple language with laymen terms if possible.
    c. Present with bullet points if that is better.
    d. You should decide on the length of the answer based on the question. If the question is very specific, then the answer should be short. If the question is general, then the answer can be long.
    e. From the false claims you found, you can mention the ones relevant to the question.
    f. If you do not know the answer or unsure about the answer, you can mention that.
"""

index_questioning_two_companies = """
Given below is text related to two companies. The data given to you is scraped data from the company website. I want you to look at the data and answer the below question.

Input format:
The data of each company starts with company name which is enclosed within triple angular brackets. The data itself is enclosed within triple backticks.

Question: {}

{}

{}

Steps to follow:
1. Read the question carefully.
2. Look at the data given.
3. Classify the question into one of the below categories:
    a. Company specific question: The question is very specific to the company. You should answer based on the data given.
    b. General question: The question is general. You can answer based on your knowledge.
    c. Mix of both: The question is mix of both. You should answer based on the data given and also based on your knowledge.
4. Check if the company is making any false claims in the data. You can use your knowledge to verify the claims.
5. Answer the question as follows:
    a. Keep the answer consise and to the point.
    b. Use simple language with laymen terms if possible.
    c. Present with bullet points if that is better.
    d. You should decide on the length of the answer based on the question. If the question is very specific, then the answer should be short. If the question is general, then the answer can be long.
    e. From the false claims you found, you can mention the ones relevant to the question.
    f. If you do not know the answer or unsure about the answer, you can mention that.

"""

index_questioning_system_prompt = """
You are a respectful chatbot assissting a user in analysing a company for investment purposes. You goal is to guide the user to make better assessments about the company. You should talk like a human. Try to avoid saying obvious stuff like "sure here is the answer" or "I hope it helps etc.". Keep the conversation natural and human like."""