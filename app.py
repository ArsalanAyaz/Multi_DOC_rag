# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_qdrant import QdrantVectorStore
# import requests

# app = FastAPI()

# # Initialize the embedding model and Gemini summarization model
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo"
# )

# Gemini = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo",
#     temperature=0,
#     max_tokens=250,
#     timeout=None,
#     max_retries=2,
# )

# # Load and prepare the document dataset (this should be done once during app startup)
# pdf_path = r"D:\Mission\PIAIC\Quarter 4\projects\Multi_DOC_rag\transformer.pdf"
# Dataset_transformer = PyPDFLoader(pdf_path)
# pages = Dataset_transformer.load()
# pdf_text = " ".join(page.page_content for page in pages)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     separators=["\n\n", ".", " "]
# )
# texts = text_splitter.create_documents([pdf_text])

# # Configure the Qdrant vector store
# url = "https://2c198102-ae27-482e-8f0c-7e7aac93309d.europe-west3-0.gcp.cloud.qdrant.io:6333"
# api_key = "MJ4SW_YT6rAEGSMffQjYO17rth77dmR_l4wq2CjehmrF9v6S9MPbLQ"

# # Check Qdrant connectivity and initialize the vector store
# response = requests.get(f"{url}/collections", headers={"api-key": api_key})
# if response.status_code != 200:
#     raise Exception(f"Failed to connect to Qdrant: {response.status_code} - {response.text}")

# qdrant = QdrantVectorStore.from_documents(
#     texts,
#     embeddings,
#     url=url,
#     api_key=api_key,
#     collection_name="Multi_rag_app",
# )


# # Define the request model for querying the API
# class QueryRequest(BaseModel):
#     question: str


# @app.post("/search")
# async def search_and_summarize(request: QueryRequest):
#     """
#     Search for documents based on a question and summarize the results.
#     """
#     question = request.question
#     docs_ss = qdrant.similarity_search(question, k=5)

#     if not docs_ss:
#         raise HTTPException(status_code=404, detail="No similar documents found.")
    
#     # Aggregate content from the top results
#     aggregated_content = " ".join([doc.page_content for doc in docs_ss])

#     # Use the Gemini model to summarize the content
#     messages = [
#         ("system", "You are a helpful assistant."),
#         ("human", f"Summarize this content: {aggregated_content[:1500]}")  # Limit content length for summarization
#     ]
    
#     # Summarize the aggregated content
#     ai_msg = Gemini.invoke(messages)

#     return {"summary": ai_msg.content}


# # To run the FastAPI application, save this as `main.py` and run:
# # uvicorn main:app --reload



# ==========================================================================


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from fastapi.responses import StreamingResponse
import requests
import asyncio
import time
from typing import AsyncGenerator

app = FastAPI()

# Initialize the embedding model and Gemini summarization model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo"
)

Gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo",
    temperature=0,
    max_tokens=250,
    timeout=None,
    max_retries=2,
)

# Load and prepare the document dataset (this should be done once during app startup)
pdf_path = r"D:\Mission\PIAIC\Quarter 4\projects\Multi_DOC_rag\transformer.pdf"
Dataset_transformer = PyPDFLoader(pdf_path)
pages = Dataset_transformer.load()
pdf_text = " ".join(page.page_content for page in pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", ".", " "]
)
texts = text_splitter.create_documents([pdf_text])

# Configure the Qdrant vector store
url = "https://2c198102-ae27-482e-8f0c-7e7aac93309d.europe-west3-0.gcp.cloud.qdrant.io:6333"
api_key = "MJ4SW_YT6rAEGSMffQjYO17rth77dmR_l4wq2CjehmrF9v6S9MPbLQ"

# Check Qdrant connectivity and initialize the vector store
response = requests.get(f"{url}/collections", headers={"api-key": api_key})
if response.status_code != 200:
    raise Exception(f"Failed to connect to Qdrant: {response.status_code} - {response.text}")

qdrant = QdrantVectorStore.from_documents(
    texts,
    embeddings,
    url=url,
    api_key=api_key,
    collection_name="Multi_rag_app",
)


# Define the request model for querying the API
class QueryRequest(BaseModel):
    question: str


# Streaming response generator
async def stream_summary(aggregated_content: str) -> AsyncGenerator[str, None]:
    # Prepare the Gemini API request for summarization
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", f"Summarize this content: {aggregated_content[:1500]}")  # Limit content length for summarization
    ]

    # Simulate the streaming response from Gemini
    ai_msg = Gemini.invoke(messages)

    # Simulating a stream of chunks by yielding parts of the content
    summary = ai_msg.content
    for i in range(0, len(summary), 100):  # Simulate streaming chunks of 100 characters
        yield summary[i:i + 100]
        await asyncio.sleep(0.1)  # Simulate network delay or processing delay


@app.post("/search")
async def search_and_summarize(request: QueryRequest):
    """
    Search for documents based on a question and summarize the results.
    """
    question = request.question
    docs_ss = qdrant.similarity_search(question, k=5)

    if not docs_ss:
        raise HTTPException(status_code=404, detail="No similar documents found.")
    
    # Aggregate content from the top results
    aggregated_content = " ".join([doc.page_content for doc in docs_ss])

    # Return a streaming response
    return StreamingResponse(stream_summary(aggregated_content), media_type="text/plain")


# To run the FastAPI application, save this as `main.py` and run:
# uvicorn main:app --reload

