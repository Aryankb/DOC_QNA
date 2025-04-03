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
## Detailed Explaination
I was working with agentic graph RAG recently. The project aimed at Document QnA from a story book. It required Agentic graph-RAG approach as in a story, different characters can have different relationships with each other. Also different incidents are linked to each other. So, here agentic graph-RAG worked well if someone asks deep questions from story. So i made graph where centre node was book --> first level nodes were chunks (each chapter of that book) --> second level nodes were atomic_facts from that chunk--> final level nodes were characters. Same characters from different atomic_facts were connected (so as to know that character was present in which parts of the story). Also different characters were connected to each other having some relationships.

So for finding the answer of user's question, I used LANGGRAPH. It was really lightweight , Fast and simple to implement it.

There were agents -

character finder (to find all characters, user's question contains)
atomic fact finder ( worked at finding the relevant atomic facts for each character, according to user query. Using similarity search)
information validator (checked if information is enough)
atomic fact extractor (if information is not enough, then extract the nearby atomic fact, or nearby events in that story)
final composer (compose final answer using extracted information and user query.
It was a fixed agentic system, worked at graphs of one kind only

I have a question, regarding what is the requirement ? Do we need to create a Interface, where user can create his own agentic system for his Graph , just using drag-drop or prompts (without code)
If yes, then using LANGGRAPH or completely doing it manually would be the best approach, because there will be transparency , and user would know what is happening. Using CrewAI would not be a good idea , as there we don't know what is happening under the hood.

FULL PIPELINE FOR KNOWLEDGE GRAPH CREATION : https://github.com/Aryankb/DOC_QNA/blob/main/graphreader_import.ipynb

FULL PIPELINE FOR RAG : https://github.com/Aryankb/DOC_QNA/blob/main/graphreader_langgraph.ipynb
