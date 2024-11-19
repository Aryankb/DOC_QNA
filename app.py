import streamlit as st
import asyncio
import os
from datetime import datetime
from hashlib import md5
from typing import Dict, List
    


import pandas as pd
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from io import BytesIO



import os
from operator import add
import re
import ast
from typing import List, Dict, Literal, Annotated
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from pydantic import BaseModel, Field


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    # other params...
)


load_dotenv()

uri=os.getenv("NEO4J_URI") 
username=os.getenv("NEO4J_USERNAME")
password=os.getenv("NEO4J_PASSWORD")
os.environ['LANGCHAIN_TRACING_V2'] ="true"
os.environ['LANGCHAIN_API_KEY'] =os.getenv("LANGCHAIN_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")    
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")




graph = Neo4jGraph(refresh_schema=False)
neo4j_graph = Neo4jGraph(refresh_schema=False)
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")


construction_system = """
You are now an intelligent assistant tasked with meticulously extracting both key elements and
atomic facts from a long text.
1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g.,
actions), and adjectives (e.g., states, feelings) that are pivotal to the text's narrative.
2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include
propositions, theories, existences, concepts, and implicit elements like logic, causality, event
sequences, interpersonal relationships, timelines, etc.
Requirements:
#####
1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
2. You should extract key elements and atomic facts comprehensively, especially those that are
important and potentially query-worthy and do not leave out details.
3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He,
She to actual names).
4. Ensure that the key elements and atomic facts you extract are presented in the same language as
the original text (e.g., English or hindi).
#####
1. Output a json dictionary. This dictionary will have a key 'atomic_facts' whose value will be a list of atomic facts.
Each element in the list should be a dictionary with keys 'key_elements' (list of key_elements)  and 'atomic_fact' (fact description).
2. keep all keys in double quotes.
"""

construction_human = """Use the given format to extract information from the 
following input: {input}"""

construction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            construction_system,
        ),
        (
            "human",
            construction_human,
        ),
    ]
)





llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

construction_chain = construction_prompt | llm






import_query = """
MERGE (d:Document {id:$document_name})
WITH d
UNWIND $data AS row
MERGE (c:Chunk {id: row.chunk_id})
SET c.text = row.chunk_text,
    c.index = row.index,
    c.document_name = row.document_name
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, row
UNWIND row.atomic_facts AS af
MERGE (a:AtomicFact {id: af.id})
SET a.text = af.atomic_fact
MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
WITH c, a, af
UNWIND af.key_elements AS ke
MERGE (k:KeyElement {id: ke})
MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
"""


graph_creation_query = """
    MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
WHERE d.id = {id:$document_name}
WITH c ORDER BY c.index WITH collect(c) AS nodes
UNWIND range(0, size(nodes) -2) AS index
WITH nodes[index] AS start, nodes[index + 1] AS end
MERGE (start)-[:NEXT]->(end)
"""




def encode_md5(text):
    return md5(text.encode("utf-8")).hexdigest()






import json
async def process_document(text, document_name, chunk_size, chunk_overlap):
    start = datetime.now()
    print(f"Started extraction at: {start}")
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    print(f"Total text chunks: {len(texts)}")
    tasks = []
    idx=0
    while idx < len(texts):
        unformatted = construction_chain.invoke({"input":text[idx]})
        # formatted_outer=json.loads(unformatted.model_dump())   
        try:
            wow=json.loads(unformatted.content[7:-4])
            # print(wow,"\n")
            tasks.append(wow)
            print(idx)
            idx+=1
        except:
            continue

        
        # print(unformatted.content,"\n")
        
    
    print(f"Finished LLM extraction after: {datetime.now() - start}")
    print(tasks)
    for index, doc in enumerate(tasks):
        doc['chunk_id'] = encode_md5(texts[index])
        doc['chunk_text'] = texts[index]
        doc['index'] = index
        for af in doc["atomic_facts"]:

            af["id"] = encode_md5(af["atomic_fact"])
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        session.run(import_query, document_name=document_name, data=tasks)

    with driver.session() as session:
        session.run(graph_creation_query, document_name=document_name)
    driver.close()
   
    


    
    print(f"Finished import at: {datetime.now() - start}")
































class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str
    analysis: str
    previous_actions: List[str]

class OverallState(TypedDict):
    question: str
    rational_plan: str
    notebook: str
    previous_actions: Annotated[List[str], add]
    check_atomic_facts_queue: List[str]
    check_chunks_queue: List[str]
    neighbor_check_queue: List[str]
    chosen_action: str
def parse_function(input_str):
    # Regular expression to capture the function name and arguments
    pattern = r'(\w+)(?:\((.*)\))?'
    
    match = re.match(pattern, input_str)
    if match:
        function_name = match.group(1)  # Extract the function name
        raw_arguments = match.group(2)  # Extract the arguments as a string        
        # If there are arguments, attempt to parse them
        arguments = []
        if raw_arguments:
            try:
                # Use ast.literal_eval to safely evaluate and convert the arguments
                parsed_args = ast.literal_eval(f'({raw_arguments})')  # Wrap in tuple parentheses
                # Ensure it's always treated as a tuple even with a single argument
                arguments = list(parsed_args) if isinstance(parsed_args, tuple) else [parsed_args]
            except (ValueError, SyntaxError):
                # In case of failure to parse, return the raw argument string
                arguments = [raw_arguments.strip()]
        

        return {
            'function_name': function_name,
            'arguments': arguments
        }
    else:
        return None

rational_plan_system = """As an intelligent assistant, your primary objective is to answer the question by gathering
supporting facts from a given article. To facilitate this objective, the first step is to make
a rational plan based on the question. This plan should outline the step-by-step process to
resolve the question and specify the key information required to formulate a comprehensive answer.
Example:
#####
User: Who had a longer tennis career, Danny or Alice?
Assistant: In order to answer this question, we first need to find the length of Danny’s
and Alice’s tennis careers, such as the start and retirement of their careers, and then compare the
two.
#####
Please strictly follow the above format. Let’s begin."""

rational_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            rational_plan_system,
        ),
        (
            "human",
            (
                "{question}"
            ),
        ),
    ]
)

rational_chain = rational_prompt | model | StrOutputParser()

def rational_plan_node(state: InputState) -> OverallState:
    rational_plan = rational_chain.invoke({"question": state.get("question")})
    print("-" * 20)
    print(f"Step: rational_plan")
    print(f"Rational plan: {rational_plan}")
    return {
        "rational_plan": rational_plan,
        "previous_actions": ["rational_plan"],
    }

neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    index_name="keyelements",
    node_label="KeyElement",
    text_node_properties=["id"],
    embedding_node_property="embedding",
    retrieval_query="RETURN node.id AS text, score, {} AS metadata"
)

def get_potential_nodes(question: str) -> List[str]:
    data = neo4j_vector.similarity_search(question, k=50)
    ret=[el.page_content for el in data]
    print(ret)
    return ret


initial_node_system = """
As an intelligent assistant, your primary objective is to answer questions based on information
contained within a text. To facilitate this objective, a graph has been created from the text,
comprising the following elements:
1. Text Chunks: Chunks of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to check a list of nodes, with the objective of selecting the most relevant initial nodes from the graph to efficiently answer the question. You are given the question, the
rational plan, and a list of node key elements. These initial nodes are crucial because they are the
starting point for searching for relevant information.
Requirements:
#####
1. Once you have selected a starting node, assess its relevance to the potential answer by assigning
a score between 0 and 100. A score of 100 implies a high likelihood of relevance to the answer,
whereas a score of 0 suggests minimal relevance.
2. Present each chosen starting node in a separate line, accompanied by its relevance score. Format
each line as follows: Node: [Key Element of Node], Score: [Relevance Score].
3. Please select at least 15 starting nodes, ensuring they are non-repetitive and diverse.
4. In the user’s input, each line constitutes a node. When selecting the starting node, please make
your choice from those provided, and refrain from fabricating your own. The nodes you output
must correspond exactly to the nodes given by the user, with identical wording.
Finally, I emphasize again that you need to select the starting node from the given Nodes, and
it must be consistent with the words of the node you selected. Please strictly follow the above
format. Return a list of dictionaries, each dictionary should have the keys : key_element and score. Keys should be strictly enclosed in double quotes. Let’s begin.
"""

initial_node_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            initial_node_system,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Nodes: {nodes}"""
            ),
        ),
    ]
)



class Node(BaseModel):
    key_element: str = Field(description="""Key element or name of a relevant node""")
    score: int = Field(description="""Relevance to the potential answer by assigning
a score between 0 and 100. A score of 100 implies a high likelihood of relevance to the answer,
whereas a score of 0 suggests minimal relevance.""")

class InitialNodes(BaseModel):
    initial_nodes: List[Node] = Field(description="List of relevant nodes to the question and plan")

initial_nodes_chain = initial_node_prompt | model


def initial_node_selection(state: OverallState) -> OverallState:
    potential_nodes = get_potential_nodes(state.get("question"))
    initial_nodes = initial_nodes_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "nodes": potential_nodes,
        }
    )
    print(initial_nodes.content)
    # paper uses 5 initial nodes
    check_atomic_facts_queue = [
        el["key_element"] for el in json.loads(initial_nodes.content[7:-4]) if el['score'] >= 50
    ]
    return {
        "check_atomic_facts_queue": check_atomic_facts_queue,
        "previous_actions": ["initial_node_selection"],
    }


atomic_fact_check_system = """As an intelligent assistant, your primary objective is to answer questions based on information
contained within a text. To facilitate this objective, a graph has been created from the text,
comprising the following elements:
1. Text Chunks: Chunks of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to check a node and its associated atomic facts, with the objective of
determining whether to proceed with reviewing the text chunk corresponding to these atomic facts.
Given the question, the rational plan, previous actions, notebook content, and the current node’s
atomic facts and their corresponding chunk IDs, you have the following Action Options:
#####
Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting nodes or chunks.
2. You can choose to read multiple text chunks at the same time.
3. Atomic facts only cover part of the information in the text chunk, so even if you feel that the
atomic facts are slightly relevant to the question, please try to read the text chunk to get more
complete information.
#####
Finally, it is emphasized again that even if the atomic fact is only slightly relevant to the
question, you should still look at the text chunk to avoid missing information. You should only
choose stop_and_read_neighbor() when you are very sure that the given text chunk is irrelevant to
the question. Please strictly follow the above format.
#####
Output a valid dictionary , dictionary should have the keys : 
"updated_notebook" (First, combine your current notebook with new insights and findings about
the question from current atomic facts, creating a more complete version of the notebook that
contains more valid information.), 
"rational_next_action" (Based on the given question, the rational plan, previous actions, and
notebook content, analyze how to choose the next action.), 
"chosen_action" (1. read_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic
fact may hold the necessary information to answer the question. This will allow you to access
more complete and detailed information.
2. stop_and_read_neighbor(): Choose this action if you ascertain that all text chunks lack valuable
information.). 
Keys should be strictly enclosed in double quotes. 
Use double quotes instead of single quotes everywhere. Let’s begin.
"""

class AtomicFactOutput(BaseModel):
    updated_notebook: str = Field(description="""First, combine your current notebook with new insights and findings about
the question from current atomic facts, creating a more complete version of the notebook that
contains more valid information.""")
    rational_next_action: str = Field(description="""Based on the given question, the rational plan, previous actions, and
notebook content, analyze how to choose the next action.""")
    chosen_action: str = Field(description="""1. read_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic
fact may hold the necessary information to answer the question. This will allow you to access
more complete and detailed information.
2. stop_and_read_neighbor(): Choose this action if you ascertain that all text chunks lack valuable
information.""")

atomic_fact_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            atomic_fact_check_system,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Previous actions: {previous_actions}
Notebook: {notebook}
Atomic facts: {atomic_facts}"""
            ),
        ),
    ]
)

atomic_fact_chain = atomic_fact_check_prompt | model

def get_atomic_facts(key_elements: List[str]) -> List[Dict[str, str]]:
    data = neo4j_graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-(fact)<-[:HAS_ATOMIC_FACT]-(chunk)
    WHERE k.id IN $key_elements
    RETURN distinct chunk.id AS chunk_id, fact.text AS text
    """, params={"key_elements": key_elements})
    return data

def get_neighbors_by_key_element(key_elements):
    print(f"Key elements: {key_elements}")
    data = neo4j_graph.query("""
    MATCH (k:KeyElement)<-[:HAS_KEY_ELEMENT]-()-[:HAS_KEY_ELEMENT]->(neighbor)
    WHERE k.id IN $key_elements AND NOT neighbor.id IN $key_elements
    WITH neighbor, count(*) AS count
    ORDER BY count DESC LIMIT 50
    RETURN collect(neighbor.id) AS possible_candidates
    """, params={"key_elements":key_elements})
    return data

def atomic_fact_check(state: OverallState) -> OverallState:
    atomic_facts = get_atomic_facts(state.get("check_atomic_facts_queue"))
    print("-" * 20)
    print(f"Step: atomic_fact_check")
    print(
        f"Reading atomic facts about: {state.get('check_atomic_facts_queue')}"
    )
    atomic_facts_results = atomic_fact_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "previous_actions": state.get("previous_actions"),
            "atomic_facts": atomic_facts,
        }
    )
    lord=json.loads(atomic_facts_results.content[7:-4])
    notebook = lord["updated_notebook"]
    print(
        f"Rational for next action after atomic check: {lord['rational_next_action']}"
    )
    chosen_action = parse_function(lord["chosen_action"])
    print(f"Chosen action: {chosen_action}")
    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "check_atomic_facts_queue": [],
        "previous_actions": [
            f"atomic_fact_check({state.get('check_atomic_facts_queue')})"
        ],
    }
    if chosen_action.get("function_name") == "stop_and_read_neighbor":
        neighbors = get_neighbors_by_key_element(
            state.get("check_atomic_facts_queue")
        )
        response["neighbor_check_queue"] = neighbors
    elif chosen_action.get("function_name") == "read_chunk":
        response["check_chunks_queue"] = chosen_action.get("arguments")[0]
    return response


chunk_read_system_prompt = """As an intelligent assistant, your primary objective is to answer questions based on information
within a text. To facilitate this objective, a graph has been created from the text, comprising the
following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to assess a specific text chunk and determine whether the available information
suffices to answer the question. Given the question, rational plan, previous actions, notebook
content, and the current text chunk, you have the following Action Options:
#####
1. search_more(): Choose this action if you think that the essential information necessary to
answer the question is still lacking.
2. read_previous_chunk(): Choose this action if you feel that the previous text chunk contains
valuable information for answering the question.
3. read_subsequent_chunk(): Choose this action if you feel that the subsequent text chunk contains
valuable information for answering the question.
4. termination(): Choose this action if you believe that the information you have currently obtained
is enough to answer the question. This will allow you to summarize the gathered information and
provide a final answer.
#####
Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
2. You can only choose one action.
#####
Please strictly follow the above format. The choosen action should contain the action name folloewd by parenthesis . example : read_subsequent_chunk().
Please give the update nodebook with all information at last. Let’s begin!
"""

class ChunkOutput(BaseModel):
    
    rational_next_move: str = Field(description="""Based on the given question, rational plan, previous actions, and
notebook content, analyze how to choose the next action.""")
    chosen_action: str = Field(description="""1. search_more(): Choose this action if you think that the essential information necessary to
answer the question is still lacking.
2. read_previous_chunk(): Choose this action if you feel that the previous text chunk contains
valuable information for answering the question.
3. read_subsequent_chunk(): Choose this action if you feel that the subsequent text chunk contains
valuable information for answering the question.
4. termination(): Choose this action if you believe that the information you have currently obtained
is enough to answer the question. This will allow you to summarize the gathered information and
provide a final answer.""")
    updated_notebook: str = Field(description="""First, combine your previous notes with new insights and findings about the
question from current text chunks, creating a more complete version of the notebook that contains
more valid information.""")

chunk_read_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            chunk_read_system_prompt,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Previous actions: {previous_actions}
Notebook: {notebook}
Chunk: {chunk}"""
            ),
        ),
    ]
)

chunk_read_chain = chunk_read_prompt | model.with_structured_output(ChunkOutput)

def get_subsequent_chunk_id(chunk):
    data = neo4j_graph.query("""
    MATCH (c:Chunk)-[:NEXT]->(next)
    WHERE c.id = $id
    RETURN next.id AS next
    """)
    return data

def get_previous_chunk_id(chunk):
    data = neo4j_graph.query("""
    MATCH (c:Chunk)<-[:NEXT]-(previous)
    WHERE c.id = $id
    RETURN previous.id AS previous
    """)
    return data

def get_chunk(chunk_id: str) -> List[Dict[str, str]]:
    data = neo4j_graph.query("""
    MATCH (c:Chunk)
    WHERE c.id = $chunk_id
    RETURN c.id AS chunk_id, c.text AS text
    """, params={"chunk_id": chunk_id})
    return data

def chunk_check(state: OverallState) -> OverallState:
    check_chunks_queue = state.get("check_chunks_queue")
    chunk_id = check_chunks_queue.pop()
    print("-" * 20)
    print(f"Step: read chunk({chunk_id})")

    chunks_text = get_chunk(chunk_id)
    read_chunk_results = chunk_read_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "previous_actions": state.get("previous_actions"),
            "chunk": chunks_text,
        }
    )

    notebook = read_chunk_results.updated_notebook
    print(
        f"Rational for next action after reading chunks: {read_chunk_results.rational_next_move}"
    )
    chosen_action = parse_function(read_chunk_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    response = {
        "notebook": notebook,
        "chosen_action": chosen_action.get("function_name"),
        "previous_actions": [f"read_chunks({chunk_id})"],
    }
    if chosen_action.get("function_name") == "read_subsequent_chunk":
        subsequent_id = get_subsequent_chunk_id(chunk_id)
        check_chunks_queue.append(subsequent_id)
    elif chosen_action.get("function_name") == "read_previous_chunk":
        previous_id = get_previous_chunk_id(chunk_id)
        check_chunks_queue.append(previous_id)
    elif chosen_action.get("function_name") == "search_more":
        # Go over to next chunk
        # Else explore neighbors
        if not check_chunks_queue:
            response["chosen_action"] = "search_neighbor"
            # Get neighbors/use vector similarity
            print(f"Neighbor rational: {read_chunk_results.rational_next_move}")
            neighbors = get_potential_nodes(
                read_chunk_results.rational_next_move
            )
            response["neighbor_check_queue"] = neighbors

    response["check_chunks_queue"] = check_chunks_queue
    return response

neighbor_select_system_prompt = """
As an intelligent assistant, your primary objective is to answer questions based on information
within a text. To facilitate this objective, a graph has been created from the text, comprising the
following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
Your current task is to assess all neighboring nodes of the current node, with the objective of determining whether to proceed to the next neighboring node. Given the question, rational
plan, previous actions, notebook content, and the neighbors of the current node, you have the
following Action Options:
#####
1. read_neighbor_node(key element of node): Choose this action if you believe that any of the
neighboring nodes may contain information relevant to the question. Note that you should focus
on one neighbor node at a time.
2. termination(): Choose this action if you believe that none of the neighboring nodes possess
information that could answer the question.
#####
Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
2. You can only choose one action. This means that you can choose to read only one neighbor
node or choose to terminate.
#####
Please strictly follow the above format. Let’s begin.
"""

class NeighborOutput(BaseModel):
    rational_next_move: str = Field(description="""Based on the given question, rational plan, previous actions, and
notebook content, analyze how to choose the next action.""")
    chosen_action: str = Field(description="""You have the following Action Options:
1. read_neighbor_node(key element of node): Choose this action if you believe that any of the
neighboring nodes may contain information relevant to the question. Note that you should focus
on one neighbor node at a time.
2. termination(): Choose this action if you believe that none of the neighboring nodes possess
information that could answer the question.""")

neighbor_select_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            neighbor_select_system_prompt,
        ),
        (
            "human",
            (
                """Question: {question}
Plan: {rational_plan}
Previous actions: {previous_actions}
Notebook: {notebook}
Neighbor nodes: {nodes}"""
            ),
        ),
    ]
)

neighbor_select_chain = neighbor_select_prompt | model.with_structured_output(NeighborOutput)

def neighbor_select(state: OverallState) -> OverallState:
    print("-" * 20)
    print(f"Step: neighbor select")
    print(f"Possible candidates: {state.get('neighbor_check_queue')}")
    neighbor_select_results = neighbor_select_chain.invoke(
        {
            "question": state.get("question"),
            "rational_plan": state.get("rational_plan"),
            "notebook": state.get("notebook"),
            "nodes": state.get("neighbor_check_queue"),
            "previous_actions": state.get("previous_actions"),
        }
    )
    print(
        f"Rational for next action after selecting neighbor: {neighbor_select_results.rational_next_move}"
    )
    chosen_action = parse_function(neighbor_select_results.chosen_action)
    print(f"Chosen action: {chosen_action}")
    # Empty neighbor select queue
    response = {
        "chosen_action": chosen_action.get("function_name"),
        "neighbor_check_queue": [],
        "previous_actions": [
            f"neighbor_select({chosen_action.get('arguments', [''])[0] if chosen_action.get('arguments', ['']) else ''})"
        ],
    }
    if chosen_action.get("function_name") == "read_neighbor_node":
        response["check_atomic_facts_queue"] = [
            chosen_action.get("arguments")[0]
        ]
    return response



answer_reasoning_system_prompt = """
As an intelligent assistant, your primary objective is to answer questions based on information
within a text. To facilitate this objective, a graph has been created from the text, comprising the
following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic
facts derived from different text chunks.
You have now explored multiple paths from various starting nodes on this graph, recording key information for each path in a notebook.
Your task now is to analyze these memories and reason to answer the question.
Strategy:
#####
1. You should first analyze each notebook content before providing a final answer.
2. During the analysis, consider complementary information from other notes and employ a
majority voting strategy to resolve any inconsistencies.
3. When generating the final answer, ensure that you take into account all available information.
#####
Example:
#####
User:
Question: Who had a longer tennis career, Danny or Alice?
Notebook of different exploration paths:
1. We only know that Danny’s tennis career started in 1972 and ended in 1990, but we don’t know
the length of Alice’s career.
2. ......
Assistant:
Analyze:
The summary of search path 1 points out that Danny’s tennis career is 1990-1972=18 years.
Although it does not indicate the length of Alice’s career, the summary of search path 2 finds this
information, that is, the length of Alice’s tennis career is 15 years. Then we can get the final
answer, that is, Danny’s tennis career is longer than Alice’s.
Final answer:
Danny’s tennis career is longer than Alice’s.
#####
Please strictly follow the above format. Let’s begin
"""

answer_reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            answer_reasoning_system_prompt,
        ),
        (
            "human",
            (
                """Question: {question}
Notebook: {notebook}"""
            ),
        ),
    ]
)

class AnswerReasonOutput(BaseModel):
    analyze: str = Field(description="""You should first analyze each notebook content before providing a final answer.
    During the analysis, consider complementary information from other notes and employ a
majority voting strategy to resolve any inconsistencies.""")
    final_answer: str = Field(description="""When generating the final answer, ensure that you take into account all available information.""")

answer_reasoning_chain = answer_reasoning_prompt | model.with_structured_output(AnswerReasonOutput)

def answer_reasoning(state: OverallState) -> OutputState:
    print("-" * 20)
    print("Step: Answer Reasoning")
    final_answer = answer_reasoning_chain.invoke(
        {"question": state.get("question"), "notebook": state.get("notebook")}
    )
    return {
        "answer": final_answer.final_answer,
        "analysis": final_answer.analyze,
        "previous_actions": ["answer_reasoning"],
    }



def atomic_fact_condition(
    state: OverallState,
) -> Literal["neighbor_select", "chunk_check"]:
    if state.get("chosen_action") == "stop_and_read_neighbor":
        return "neighbor_select"
    elif state.get("chosen_action") == "read_chunk":
        return "chunk_check"

def chunk_condition(
    state: OverallState,
) -> Literal["answer_reasoning", "chunk_check", "neighbor_select"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") in ["read_subsequent_chunk", "read_previous_chunk", "search_more"]:
        return "chunk_check"
    elif state.get("chosen_action") == "search_neighbor":
        return "neighbor_select"

def neighbor_condition(
    state: OverallState,
) -> Literal["answer_reasoning", "atomic_fact_check"]:
    if state.get("chosen_action") == "termination":
        return "answer_reasoning"
    elif state.get("chosen_action") == "read_neighbor_node":
        return "atomic_fact_check"



langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
langgraph.add_node(rational_plan_node)
langgraph.add_node(initial_node_selection)
langgraph.add_node(atomic_fact_check)
langgraph.add_node(chunk_check)
langgraph.add_node(answer_reasoning)
langgraph.add_node(neighbor_select)

langgraph.add_edge(START, "rational_plan_node")
langgraph.add_edge("rational_plan_node", "initial_node_selection")
langgraph.add_edge("initial_node_selection", "atomic_fact_check")
langgraph.add_conditional_edges(
    "atomic_fact_check",
    atomic_fact_condition,
)
langgraph.add_conditional_edges(
    "chunk_check",
    chunk_condition,
)
langgraph.add_conditional_edges(
    "neighbor_select",
    neighbor_condition,
)
langgraph.add_edge("answer_reasoning", END)

langgraph = langgraph.compile()

# View
display(Image(langgraph.get_graph().draw_mermaid_png()))




async def get_answer(question: str):
    # result = await llm.ainvoke({"input": question, "document_name": document_name})
    # return result.dict()
    ans= langgraph.invoke({"question":question})
    return ans


# Set page configuration
st.set_page_config(page_title="Document Q/A Chatbot", layout="wide")

# Title of the application
st.title("📚 Document Question-Answering Chatbot")


# Step 1: Upload document (PDF or Text)
st.subheader("Upload your Document")
name=st.text_input("Enter the name of the document")
uploaded_file = st.file_uploader("Upload a PDF / CSV / TXT file", type=["pdf", "txt","csv"])

# Step 2: Input Question
if st.button("Create Knowledge Graph") and uploaded_file:   
    # Handle PDF files
    if uploaded_file.type == "application/pdf":
        # Open the PDF file using a binary stream
        binary_stream = BytesIO(uploaded_file.read())
        pdf_reader = PdfReader(binary_stream)
        FULL_TEXT = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            
            text = page.extracted_text()
            FULL_TEXT += text
            FULL_TEXT += "\npage_break\n"
        
        st.success("PDF Document uploaded successfully!")
        st.text("Creating graph from PDF content...")
    
    # Handle CSV files
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        FULL_TEXT = df.to_string()  # Convert DataFrame to string for text processing
        st.success("CSV Document uploaded successfully!")
        st.text("Creating graph from CSV content...")
    
    # Handle TXT files
    elif uploaded_file.type == "text/plain":
        FULL_TEXT = uploaded_file.read().decode("utf-8")  # Decode the byte stream to text
        st.success("TXT Document uploaded successfully!")
        st.text("Creating graph from TXT content...")
    async def call():
        
        await process_document(FULL_TEXT, name, chunk_size=1500, chunk_overlap=100)

    asyncio.run(call())
    # Process the document (you'll need to define this function)


    st.text("Graph created successfully!")

st.subheader("Ask a Question")

question = st.text_input("Type your question here...")

# Button to trigger question-answering
if st.button("Get Answer"):
    async def call2():
        
        return await get_answer(question)
    st.write(asyncio.run(call2()))


# Styling to make the interface more appealing
st.markdown("""
    <style>
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1r6slb0 {
        font-size: 1.2rem;
    }
    .stButton > button {
        background-color: #008CBA;
        border: none;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

