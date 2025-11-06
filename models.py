import pydantic
from typing import Optional

class RAGChunkAndSource(pydantic.BaseModel):
    chunks: list[str]
    source_id: Optional[str] = None

class RAGUpsertRequest(pydantic.BaseModel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: set[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: set[str]
    num_contexts: int