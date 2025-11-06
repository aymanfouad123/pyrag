import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime

from openai import OpenAI
from data_loader import load_data, embed_chunks
from vector_db import QdrantStorage
from models import RAGUpsertRequest, RAGSearchResult, RAGQueryResult, RAGChunkAndSource

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(fn_id="RAG Ingest PDF", trigger=inngest.TriggerEvent(event="rag/ingest_pdf"))
async def rag_ingest_pdf(ctx: inngest.Context):
    def load_and_chunk():
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_data(pdf_path)
        return {"chunks": chunks, "source_id": source_id}
    
    def upsert():
        chunks = chunks_and_source["chunks"]
        source_id = chunks_and_source["source_id"]
        vectors = embed_chunks(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}-{i}")) for i in range(len(chunks))]
        payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
        QdrantStorage().add_vectors(ids, vectors, payloads)
        return {"ingested": len(chunks)}
    
    chunks_and_source = await ctx.step.run("load-and-chunk", load_and_chunk)
    ingested = await ctx.step.run("upsert", upsert)
    
    return ingested

@inngest_client.create_function(fn_id="RAG Query", trigger=inngest.TriggerEvent(event="rag/query"))
async def rag_query(ctx: inngest.Context):
    
    # Step 1: Embed and search
    def embed_and_search():
        question = ctx.event.data["question"]
        top_k = ctx.event.data.get("top_k", 5)
        query_vector = embed_chunks([question])[0]
        query_result = QdrantStorage().search(query_vector, top_k)
        return {
            "contexts": query_result["context"],
            "sources": list(query_result["sources"])
        }
    
    # Step 2: Generate answer
    def generate_answer():
        contexts = search_result["contexts"]
        sources = search_result["sources"]
        question = ctx.event.data["question"]
        
        context_block = "\n\n".join(contexts)
        user_content = (
            "You are a helpful assistant that can answer questions about the following context:\n\n"
            f"{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely using the context provided."
        )
        
        return {
            "context_block": context_block,
            "user_content": user_content,
            "sources": sources,
            "num_contexts": len(contexts)
        }
    
    search_result = await ctx.step.run("embed-and-search", embed_and_search)
    llm_input = await ctx.step.run("prepare-llm-input", generate_answer)
    
    # Step 3: Call LLM
    def call_llm():
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based on provided context."
                },
                {
                    "role": "user", 
                    "content": llm_input["user_content"]
                }
            ],
            max_tokens=1024,
            temperature=0.2
        )
        
        return {
            "answer": response.choices[0].message.content.strip()
        }

    llm_result = await ctx.step.run("llm-answer", call_llm)
    answer = llm_result["answer"]
        
    return {
        "answer": answer,
        "sources": llm_input["sources"],
        "num_contexts": llm_input["num_contexts"]
    }
    
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query])