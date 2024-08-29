import chainlit as cl

from dotenv import load_dotenv # type: ignore
import os
import openai

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain.schema.runnable.config import RunnableConfig
from langchain.tools.retriever import create_retriever_tool

from yfiles_jupyter_graphs import GraphWidget
from neo4j import GraphDatabase

from PyPDF2 import PdfReader
from langchain.schema import Document

from typing import Tuple, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer

#getting chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, AIMessage

import logging

# Warning control
import warnings
warnings.filterwarnings("ignore")

from io import BytesIO


from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

load_dotenv('.env', override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MODEL = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MODEL')
NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MODEL,
    api_version=AZURE_OPENAI_API_VERSION,
    api_key=OPENAI_API_KEY,
    temperature=0.3
)
# graph = Neo4jGraph()
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

def read_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    documents = []
    
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        metadata = {
            "source": pdf_path,
            "page": page_num + 1 
        }
        documents.append(Document(page_content=text, metadata=metadata))
    
    return documents


def data_ingestion(pdf_path):
    try:
    
        raw_document = read_pdf(pdf_path)
        
        from langchain.text_splitter import TokenTextSplitter
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        documents = text_splitter.split_documents(raw_document[:3])
        
        llm_transformer = LLMGraphTransformer(llm=llm)
        
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        return "data ingestion is successfull"
       
    except:
        return "Data ingestion is fail"
    
    
    

@cl.on_chat_start
async def main():
    # content = ""
    # await cl.Message(content=content).send()
    
    # files = None

    # # Wait for the user to upload a file
    # while files is None:
    #     files = await cl.AskFileMessage(
    #         content="Please upload a pdf file to begin!",
    #         accept=["application/pdf"],
    #         max_size_mb=20,
    #         timeout=180,
    #     ).send()

    # text_file = files[0]
    
    # pdf_read = BytesIO(text_file.path)
    # msg = cl.Message(content="You can ask any question..")
    # await msg.send()
    
    
    # data_ingestion(text_file.path)
    
    
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002",
                                    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                                    api_key=OPENAI_API_KEY
                                    )

    vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        index_name='neo4j'
    )
    
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    from langchain_core.pydantic_v1 import BaseModel, Field
    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
            "appear in the text",
        )
        
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    entity_chain = prompt | llm.with_structured_output(Entities)
    
    def generate_full_text_query(input: str) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    
    def structured_retriever(question: str) -> str:
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 100
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    
    
    # def retriever(question: str):
    #     print(f"Search query: {question}")
    #     structured_data = structured_retriever(question)
    #     unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    #     final_data = f"""Structured data:
    #     {structured_data}
    #     Unstructured data:
    #     {"#Document ". join(unstructured_data)}
    #     """
    #     return final_data
    
    
    def retriever(question: str):
        print(f"Search query: {question} (type: {type(question)})")
        
        # Ensure the question is a string before proceeding
        if not isinstance(question, str):
            question = str(question)  # Convert it to string if it's not already
        
        structured_data = structured_retriever(question)
        
        # Check if similarity_search expects a string and handle accordingly
        unstructured_data = [
            el.page_content for el in vector_index.similarity_search(question)
        ]
        
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ".join(unstructured_data)}
        """
        return final_data
    
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    
    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ), 
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x : x["question"]),
    )
    
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    
    chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # msg.content = f"Processing {text_file.name} done. You can now ask questions!"
    # await msg.update()
    
    cl.user_session.set("chain", chain)
  

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  

    msg = cl.Message(content="")

    async for chunk in chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    
