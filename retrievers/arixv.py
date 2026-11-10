from langchain_community.retrievers import ArxivRetriever

#create retriver
retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

#qury arixv
docs=retriever.invoke("Explain AI")

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:")
    print("Title:",doc.metadata.get("title"))
    print("Authors", doc.metadata.get("authors"))
    print("Summary:", doc.page_content[:500])  # Print the first 500 characters of the summary