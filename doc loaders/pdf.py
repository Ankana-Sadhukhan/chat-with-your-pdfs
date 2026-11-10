from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)


data= PyPDFLoader("doc loaders/AI-rag.pdf")
docs=data.load()
texts = text_splitter.split_documents(docs)
print(texts[1].page_content)