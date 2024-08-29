from dotenv import load_dotenv # type: ignore
import os
# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import AzureChatOpenAI
from PyPDF2 import PdfReader
from langchain.schema import Document

from langchain_experimental.graph_transformers import LLMGraphTransformer

# Warning control
import warnings
warnings.filterwarnings("ignore")


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
    
# Directory containing the PDFs
pdf_folder = "pdf"

# List all files in the pdf folder
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

for file in pdf_files:
    data_ingestion(file)
    print(f'{os.path.basename(file)} file successfully added to knowledge graph...')
    
print("All the documents successfully uploaded!")