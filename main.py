import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
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
    def embed_and_search(question: str, top_k: int = 5):
        query_vector = embed_chunks([question])[0]
        query_result = QdrantStorage().search(query_vector, top_k)
        return RAGSearchResult(contexts=query_result["context"], sources=query_result["sources"])

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)
    query_result = await ctx.step.run("embed-and-search", lambda: embed_and_search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(query_result.contexts)
    user_content = (
    "You are a helpful assistant that can answer questions about the following context:"
    f"{context_block}"
    f"Question: {question}"
    "Answer concisely using the context provided."
    )
    
    adapter = ai.openai.Adapter(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    res = await ctx.step.ai.infer("llm-answer", adapter, body ={"max_tokens": 1024, "temperature": 0.2, "messages": [{"role": "system", "content": "You are a helpful assistant that can answer questions about the following context:"}, {"role": "user", "content": user_content}]})
    answer = res.choices[0].message.content.strip()
    return RAGQueryResult(answer=answer, sources=query_result.sources, num_contexts=len(query_result.contexts))
    
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query])

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}