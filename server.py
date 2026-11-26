import os
import re
import uuid
import httpx
import pymupdf 
import arxiv
from fastmcp import FastMCP 

# Qdrant client imports
from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer

# Modern Markdown & Layout Splitter imports
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. Initialize FastMCP Server
mcp = FastMCP("Personal Research Analyst")

# 2. Connect to Docker Qdrant Server & Local Embedding Model
qdrant_client = AsyncQdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "research_papers"
encoder = SentenceTransformer("all-MiniLM-L6-v2")

async def init_qdrant():
    """Ensure the target collection exists in the Docker instance with correct vectors config."""
    if not await qdrant_client.collection_exists(COLLECTION_NAME):
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,  
                distance=models.Distance.COSINE
            )
        )

async def download_file(url: str, save_path: str):
    """Helper function for downloading files over HTTP."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)

# ---------------------------------------------------------------------------
# MCP TOOL 1: Search arXiv for research papers
# ---------------------------------------------------------------------------
@mcp.tool()
async def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for scientific papers by keyword.
    Returns paper titles, summaries, and downloadable direct PDF URLs.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in client.results(search):
            pdf_url = result.pdf_url if result.pdf_url else "No direct PDF link found"
            results.append(f"📄 **Title:** {result.title}\n🔗 **PDF URL:** {pdf_url}\n")
            
        return "\n".join(results) if results else "No papers found for this query."
    except Exception as e:
        return f"Error connecting to arXiv API: {str(e)}"

# ---------------------------------------------------------------------------
# MCP TOOL 2: Layout-Aware Parsing & Storage
# ---------------------------------------------------------------------------
@mcp.tool()
async def ingest_and_vectorize(pdf_url: str) -> str:
    """
    Downloads a PDF, extracts its layout structure as Markdown (preserving tables/headers),
    splits it hierarchically by document sections, and stores them in Qdrant.
    """
    await init_qdrant()
    local_filename = f"temp_{uuid.uuid4().hex}.pdf"
    
    try:
        # Download PDF
        await download_file(pdf_url, local_filename)
        
        # LAYOUT EXTRACTION: Convert PDF directly to rich Markdown text strings
        md_text = pymupdf4llm.to_markdown(local_filename)
        os.remove(local_filename)
        
        if not md_text.strip():
            return "Failed to extract layout-aware structural text from the PDF."
            
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        
        # 1. First Pass: Split document by its physical logical headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(md_text)
        
        # 2. Second Pass: Sub-split massive sections so they don't break LLM context limits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_docs = text_splitter.split_documents(header_splits)
        
        chunks = []
        metadatas = []
        for doc in final_docs:
            h1 = doc.metadata.get("Header_1", "")
            h2 = doc.metadata.get("Header_2", "")
            h3 = doc.metadata.get("Header_3", "")
            context_prefix = f"Section: {h1} > {h2} > {h3}\n\n" if h1 else ""
            
            chunks.append(context_prefix + doc.page_content)
            metadatas.append({
                "source": pdf_url,
                "section": f"{h1} -> {h2}".strip(" -> ")
            })
            
        if not chunks:
            return "Layout chunking generated 0 blocks."
            
        # Generate raw embeddings locally
        embeddings = encoder.encode(chunks).tolist()
        
        # Build upsert point structures for Qdrant
        points = []
        for i, (chunk, vector, meta) in enumerate(zip(chunks, embeddings, metadatas)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),  
                    vector=vector,
                    payload={"text": chunk, "source": meta["source"], "section": meta["section"]}  
                )
            )
        
        # Batch insert directly into Qdrant Docker instance
        await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        return f"Successfully processed paper using LAYOUT-AWARE CHUNKING. Stored {len(chunks)} sections inside Docker Qdrant."
        
    except Exception as e:
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return f"Error during layout ingestion pipeline: {str(e)}"

# ---------------------------------------------------------------------------
# MCP TOOL 3: Query the Docker Qdrant DB
# ---------------------------------------------------------------------------
@mcp.tool()
async def query_vector_db(query: str, n_results: int = 2) -> str:
    """
    Queries the running Qdrant Docker container using semantic similarity search.
    Returns the top matching text chunks found across all ingested documents.
    """
    await init_qdrant()
    
    # Embed the search query
    query_vector = encoder.encode(query).tolist()
    
    # Query Qdrant using the modern query_points method
    response = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,   
        limit=n_results
    )
    
    output = []
    if response and hasattr(response, 'points') and response.points:
        for idx, hit in enumerate(response.points):
            source = hit.payload.get("source", "Unknown Source")
            section = hit.payload.get("section", "General Context")
            text_content = hit.payload.get("text", "No text context found")
            output.append(f"🔍 [Match {idx+1}] (Source: {source} | Section: {section} | Score: {hit.score:.4f}):\n{text_content}\n---")
        return "\n\n".join(output)
    
    return "No matching records found in Qdrant. Ensure you have ingested a document first."

if __name__ == "__main__":
    import logging
    # Force background loggers to stay silent so stdio doesn't corrupt json-rpc formats
    logging.getLogger('fastmcp').setLevel(logging.ERROR)
    mcp.run(transport="stdio")
