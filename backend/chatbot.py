from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import GROQ_API_KEY

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file or set the environment variable.")

client = Groq(api_key=GROQ_API_KEY)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the FAISS database that was created by vectore_store.py
db = FAISS.load_local("db", embedding, allow_dangerous_deserialization=True)

SYSTEM_PROMPT = """
You are Prajwal's AI portfolio assistant.

You can ONLY answer questions about:
- Prajwal's skills
- Prajwal's projects
- Prajwal's experience
- Prajwal's certifications

If the question is unrelated respond:

"I can only answer questions about Prajwal's portfolio."
"""

def ask_bot(question):

    docs = db.similarity_search(question)

    if len(docs) == 0:
        return "I can only answer questions about Prajwal's portfolio."

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    {SYSTEM_PROMPT}

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
    )

    return response.choices[0].message.content