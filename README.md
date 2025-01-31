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
