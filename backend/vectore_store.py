from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

files = [
    os.path.join(project_root, "data/about.txt"),
    os.path.join(project_root, "data/skills.txt"),
    os.path.join(project_root, "data/projects.txt"),
    os.path.join(project_root, "data/experience.txt")
]

documents = []

for file in files:
    loader = TextLoader(file)
    documents.extend(loader.load())

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Use FAISS instead of Chroma to avoid Pydantic v1 compatibility issues
db = FAISS.from_documents(documents, embedding)

# Save to local folder
db.save_local("db")

print("Vector database created successfully with FAISS")