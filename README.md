# Conversational Querying and Comparative Analysis of Company Portals Using LLM

## 📌 Overview
This project implements an LLM-based system that enables conversational querying of company portals, comparative analysis between two portals, and PDF-based querying.

## 🎯 Features
- **Conversational Querying**: Users can interact with company portals in a natural, question-answering format.
- **Comparative Analysis**: Compare two company portals to retrieve similarities and differences.
- **PDF Querying**: Extract key insights from financial reports and other PDF documents.
- **Two-Level Chunking for Semantic Search**: Implements parent-child chunk relationships for enhanced retrieval accuracy.
- **Query Enhancement**: Uses chat history and Google Gemini for improving query precision.

## 🔧 Tech Stack
- **Data Scraping**: `Selenium`, `Scrapy`
- **Text Processing**: `SpaCy`, `Sentence Segmentation`
- **Embedding Model**: `Google Embedding Model`
- **Vector Storage**: Dictionary containing chunks, embeddings, and relationships
- **LLM API**: `OpenAI API`
- **Semantic Search**: Top-K chunk retrieval with scoring-based selection

## ⚙️ System Workflow
1. **User Query** → Query Enhancement using Chat History  
2. **Semantic Search** → Retrieves relevant indexed data  
3. **LLM Response** → Provides a context-aware answer  
4. **Comparative Analysis** (if applicable) → Highlights differences & similarities  
5. **PDF Querying** (if applicable) → Extracts key insights from documents

## 🚧 Challenges

### 🕵️‍♂️ Scraping:
- **Old Data** 📅  
  - ✅ Checking timestamps in HTML response  
  - 🤖 Using LLaMA to determine if webpage content is outdated  

### 🧹 Scrape Response Cleaning:
- Remove unwanted URLs from prefixes (**Done** ✅)  
- Obtain latest and investment-related URLs from remaining URLs (**Done** ✅)  

### 📑 Index Creation:
- Remove duplicate paragraphs using similarity scores (**Threshold: > 0.99**)  

### ✂️ Chunking Web Page Text:
- Strictly enforce maximum paragraph size  

### 🚀 Embedding Optimization:
- Provide **GPU compatibility**  
- Pass **entire list of paragraphs** from scraped data instead of processing one-by-one  

### 💬 Improving Conversation on Index:
- Ignore chunks under a certain similarity score  
- Add **previous model responses** to the prompt  
- Modify prompts to instruct the model to consider past responses  
- Keep **only the last 5 responses** in conversation  
- Use **Google's embedding model**  
- Try **AQC-type** of search instead of embeddings  
- Consider using **cluster-level summaries**?  
- Compare indexed conversations **with and without cleaning**  
- Use **previous responses** for query enhancement  
- Ensure **only company-related** statements/phrases are used in query enhancement  


