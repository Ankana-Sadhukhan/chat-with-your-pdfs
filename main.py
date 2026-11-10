from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma-db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5}
)

llm=init_chat_model("groq:openai/gpt-oss-20b")

# prompt Template

prompt=ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Ai assistany.
Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
    """),
    ("human", """Context:
{context}
Question:
{question}""")
])

print("Rag system is created")
print("press 0 to exit")

while True:
    query=input("\nEnter your question: ")
    if query=="0":
        break
    docs=retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    final_prompt=prompt.invoke({"context":context,"question":query})
    response=llm.invoke(final_prompt)
    print(f"\nAI: {response.content}")
