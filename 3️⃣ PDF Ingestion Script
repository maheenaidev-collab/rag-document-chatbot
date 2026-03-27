from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

loader = PyPDFLoader("data/document.pdf")
docs = loader.load()

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)

db.save_local("embeddings")
