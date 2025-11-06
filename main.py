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
    
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}