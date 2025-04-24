from fastapi import FastAPI, UploadFile, Form, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import GoogleCloudEnterpriseSearchRetriever
from chain import Chain
import os
import tempfile

app = FastAPI()

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory & object instance
chat_memory = {}
chain_obj = Chain()

class QueryRequest(BaseModel):
    session_id: str
    question: str
    knowledge_base: str = "existing"# "uploaded", "existing", or "none"


@app.on_event("startup")
async def load_model():
    await chain_obj.initialize()

    
@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(...)):
    contents = await file.read()
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(temp_path.name, "wb") as f:
        f.write(contents)

    loader = PyPDFLoader(temp_path.name)
    docs = loader.load()
    os.remove(temp_path.name)

    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=50, separator=" ")
    docs_split = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")) 
    vector_store = FAISS.from_documents(docs_split, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}, search_type="mmr")

    chat_memory[session_id] = {
        "retriever": retriever,
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    }

    return {"message": "Document uploaded and retriever created successfully."}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    session_id = request.session_id
    question = request.question
    kb = request.knowledge_base

    if session_id not in chat_memory:
        chat_memory[session_id] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        }

    memory = chat_memory[session_id]["memory"]

    if kb == "existing":
        retriever = GoogleCloudEnterpriseSearchRetriever(
            project_id=os.getenv("PROJECT"),
            search_engine_id="oil-and-gas_1735900714499",
            max_documents=7,
            max_extractive_answer_count=5,
        )
    elif kb == "uploaded":
        retriever = chat_memory[session_id].get("retriever")
        if retriever is None:
            return {"error": "No uploaded document retriever found for this session."}
    else:
        retriever = None

    response = chain_obj.chain_response(question=question, memory=memory, retriever=retriever)
    memory.save_context({"input": question}, {"output": response})

    return {"response": response, "chat_history": memory.load_memory_variables({})}

@app.post("/reset/")
async def reset_memory(session_id: str):
    chat_memory[session_id] = {
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    }
    return {"message": "Memory reset for session: " + session_id}
