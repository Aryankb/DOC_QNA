# Project Title: Graph RAG with Neo4j and AI Agents

## Overview
This project implements a **Graph RAG** (Retrieval-Augmented Generation) approach to retain context from long documents, save the information in a **Neo4j** graph database, and use **AI agents** to explore and retrieve relevant information from the graph. This method improves upon the traditional RAG by maintaining context from various parts of lengthy documents, providing better retrieval and exploration of knowledge stored in graph form.


## Key Features
1. **Graph-based Information Retention**: Unlike standard RAG approaches, which may lose context over long documents, this graph-based approach retains detailed context from different sections of the document.
   
2. **Neo4j Integration**: Information from the document is structured and stored as nodes and relationships in a Neo4j graph database, allowing for efficient retrieval and exploration of interconnected concepts.

3. **AI Agents for Exploration**: AI agents interact with the graph to retrieve, explore, and infer information from the nodes and relationships, providing relevant answers to user queries.

## Workflow
1. **Document Processing**: Large PDF documents are processed to extract text and segment it into atomic facts or knowledge units.
   
3. **Graph Construction**: The extracted knowledge is stored in Neo4j as a graph. Each node represents a fact, and relationships between nodes represent logical connections, references, or context between sections.

4. **AI Agent Interaction**: AI agents explore the graph by traversing nodes and edges, retrieving relevant facts and their context. This approach ensures that the system retains important contextual information across different sections of the document.

FULL PIPELINE FOR KNOWLEDGE GRAPH CREATION : https://github.com/Aryankb/DOC_QNA/blob/main/graphreader_import.ipynb

FULL PIPELINE FOR RAG : https://github.com/Aryankb/DOC_QNA/blob/main/graphreader_langgraph.ipynb
