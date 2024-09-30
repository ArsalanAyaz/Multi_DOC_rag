# from fastapi import FastAPI


# app : FastAPI= FastAPI()


# @app.get("/")
# def greet():
#     return {"message": f"Hello, Arsalan!"}



# ----------------------------------------------------------------
# Gemini Model

# user_query = input("Enter your querry : ")


# from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo",
#     temperature=0,
#     max_tokens=250,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
# messages = [
#     (
#         "system",
#         "You are a helpful assistant",
#     ),
#     ("human", user_query),
# ]
# ai_msg = Gemini.invoke(messages)
# print(ai_msg.content)





# ----------------------------------------------------------------
# HuggingFace Models 


# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     "HuggingFaceH4/zephyr-7b-beta",
#     token="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",
# )

# for message in client.chat_completion(
# 	messages=[{"role": "user", "content": "What is allium cepa?"}],
# 	max_tokens=500,
# 	stream=True,
# ):
#     print(message.choices[0].delta.content, end="")

#-----------

# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     "microsoft/Phi-3-mini-4k-instruct",
#     token="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",
# )

# for message in client.chat_completion(
# 	messages=[{"role": "user", "content": "What is river chenab?"}],
# 	max_tokens=500,
# 	stream=True,
# ):
#     print(message.choices[0].delta.content, end="")


# -----------------------------------------------------------------------------------------    



# Gemini embedding model

# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo")
# # vector = embeddings.embed_query("hello, world!")
# # print(vector[:5])
# print(f"Total vectors = {len(vector)}")




# build rag application using three datasets

# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import PyPDFLoader

# # dataset 1
# Dataset_transformer = PyPDFLoader(r"D:\Mission\PIAIC\Quarter 4\projects\Multi_DOC_rag\transformer.pdf")
# pages = Dataset_transformer.load()
# # pdf_text = [page.page_content for page in pages]  # Extract the content of each page
# pdf_text = " ".join(page.page_content for page in pages)

# # Accessing specific pages
# print("page 1 ")
# print(pages[0])  # First page
# print("page 2 ")
# print(pages[1])  # Second page


# dataset 2
# Dataset_sennoFlush = PyPDFLoader(r"D:\Mission\PIAIC\Quarter 4\projects\Multi_DOC_rag\SennoFlush - VSL Script.pdf")
# pages =Dataset_sennoFlush.load()

# # Accessing specific pages
# print("page 1 ")
# print(pages[0])  # First page
# print("page 2 ")
# print(pages[1])  # Second page



# dataset 3
# Dataset_Alamo_cult=PyPDFLoader(r"D:\Mission\PIAIC\Quarter 4\projects\Multi_DOC_rag\Alamo Cult.pdf")
# pages =Dataset_Alamo_cult.load()

# # Accessing specific pages

# print("page 1 ")
# print(pages[0])  # First page
# print("page 2 ")
# print(pages[1])  # Second page


#----------------------------------------------------------------
# Splitters

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     # length_function=len,
#     # is_separator_regex=False,
# )
# texts = text_splitter.create_documents(pdf_text)
# print(texts[0])
# print(texts[1])
# print(texts[2])
# print(texts[3])
# print(texts[4])



# -------------------------------------------------------------------------------------------------------------------
# Database

# from langchain_qdrant import QdrantVectorStore

# url = "https://2c198102-ae27-482e-8f0c-7e7aac93309d.europe-west3-0.gcp.cloud.qdrant.io:6333"
# api_key = "MJ4SW_YT6rAEGSMffQjYO17rth77dmR_l4wq2CjehmrF9v6S9MPbLQ"
# qdrant = QdrantVectorStore.from_documents(
#     texts,
#     embeddings,
#     url=url,
#     # prefer_grpc=True,
#     api_key=api_key,
#     collection_name="Multi_rag_app",
# )

# # Searching for similar documents to a given question
# question = "What is tranformer?"
# docs_ss = qdrant.similarity_search(question,k=5)
# print(docs_ss)



# =========================================================================================


# Gemini embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo"
)

# Initialize the chat model for summarization (Google Gemini)
Gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key="AIzaSyB2J3yWoGz_UhMoDQDmOulaWzzSO1h9kZo",
    temperature=0,
    max_tokens=250,
    timeout=None,
    max_retries=2,
)

# Updated PyPDFLoader import as per the deprecation warning
from langchain_community.document_loaders import PyPDFLoader

# Load the transformer dataset PDF
Dataset_transformer = PyPDFLoader(r"D:\Mission\PIAIC\Quarter 4\projects\Multi_DOC_rag\transformer.pdf")
pages = Dataset_transformer.load()

# Verify if all pages are being loaded
print(f"Total number of pages loaded: {len(pages)}")  # Check the number of pages loaded

# Join the text from all pages into one string
pdf_text = " ".join(page.page_content for page in pages)

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Using a moderate chunk size to ensure meaningful chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Larger chunk size to avoid splitting into single characters
    chunk_overlap=200,  # Set overlap to capture context between chunks
    separators=["\n\n", ".", " "]  # Use logical text separators: paragraphs, sentences, words
)
texts = text_splitter.create_documents([pdf_text])  # Pass text as a list of documents

# Database
from langchain_qdrant import QdrantVectorStore
import requests

# Qdrant configuration
url = "https://2c198102-ae27-482e-8f0c-7e7aac93309d.europe-west3-0.gcp.cloud.qdrant.io:6333"
api_key = "MJ4SW_YT6rAEGSMffQjYO17rth77dmR_l4wq2CjehmrF9v6S9MPbLQ"

# Check if the Qdrant instance is reachable
response = requests.get(f"{url}/collections", headers={"api-key": api_key})
if response.status_code == 200:
    print("Connected to Qdrant successfully.")
else:
    print(f"Failed to connect to Qdrant: {response.status_code} - {response.text}")

# Initialize Qdrant vector store
qdrant = QdrantVectorStore.from_documents(
    texts,
    embeddings,
    url=url,
    api_key=api_key,
    collection_name="Multi_rag_app",
)

# Searching for similar documents to a given question
question = "Scaled Dot-Product Attention"
docs_ss = qdrant.similarity_search(question, k=5)  # Return top 5 results

# Check if there are search results
if docs_ss:
    # Aggregate content from the top results
    aggregated_content = " ".join([doc.page_content for doc in docs_ss])

    # Use the Gemini model to summarize the content
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", f"Summarize this content: {aggregated_content[:1500]}")  # Limit content length for summarization
    ]
    
    # Summarize the aggregated content
    ai_msg = Gemini.invoke(messages)
    
    # Directly print the summary provided by the Gemini model
    print(f"Summary:\n{ai_msg.content}")
else:
    print("No similar documents found.")
