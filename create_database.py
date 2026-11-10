import math
import re
import time

#load pdf
#split into chunks
#create the embeddings
#store in choma db
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
data= PyPDFLoader("doc loaders/AI-rag.pdf")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma-db"
)

batch_size = 100
max_attempts = 5

for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]
    for attempt in range(1, max_attempts + 1):
        try:
            vectorstore.add_documents(batch)
            break
        except GoogleGenerativeAIError as error:
            message = str(error)
            lower_message = message.lower()
            is_quota_error = (
                ("429" in message or "resource_exhausted" in lower_message)
                and ("quota" in lower_message or "per minute" in lower_message)
            )
            if not is_quota_error or attempt == max_attempts:
                raise

            delays = re.findall(
                r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s",
                message,
            )
            delays.extend(
                re.findall(r"retry in (\d+(?:\.\d+)?)s", lower_message)
            )
            retry_seconds = (
                math.ceil(max(map(float, delays))) + 1 if delays else 60
            )
            print(f"Embedding quota reached; retrying in {retry_seconds} seconds.")
            time.sleep(retry_seconds)
