from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_data(file_path: str):
    reader = PDFReader()
    documents = reader.load_data(file=file_path)
    nodes = splitter.get_nodes_from_documents(documents)
    chunks = [node.get_content() for node in nodes]
    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=chunks,
        model=EMBEDDING_MODEL
    )
    return [r.embedding for r in response.data]