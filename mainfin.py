
import os
import tempfile
import logging
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import PyPDF2

# Load environment variables from .env file manually
def load_env_file():
    """Load environment variables from .env file without python-dotenv dependency."""
    try:
        if os.path.exists('.env'):
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open('.env', 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    break
                except UnicodeDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

load_env_file()

# Fallback: Set API key directly if not loaded from .env
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBzZSJ2b60YLpcSF_Job0D__rMwbcCZS8g"

# Set your expected API key for authentication
EXPECTED_API_KEY = os.getenv("API_KEY", "1234")

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="HackRx RAG API",
    description="Process PDF documents from URLs and answer multiple questions.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token."""
    if credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Request/Response Models ---
class HackRxRequest(BaseModel):
    documents: HttpUrl  # URL to the PDF document
    questions: List[str]  # List of questions to ask

class HackRxResponse(BaseModel):
    answers: List[str]

# --- RAG System Class ---
class RAGSystem:
    def __init__(self):
        self.embedding = None
        self.layer1_retriever = None
        self.llm = None
        self.initialized = False

    def initialize_system(self, gemini_api_key: str, layer1_db_path: str = "faiss_layer1_db"):
        try:
            logger.info("🚀 Initializing RAG system...")
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

            logger.info("  - Loading embedding model...")
            self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            if os.path.exists(layer1_db_path):
                logger.info(f"  - Loading Layer 1 DB from {layer1_db_path}...")
                db = FAISS.load_local(layer1_db_path, self.embedding, allow_dangerous_deserialization=True)
                docs = list(db.docstore._dict.values())
                faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 5
                self.layer1_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever],
                    weights=[0.5, 0.5]
                )
                logger.info("  - Layer 1 hybrid retriever created.")
            else:
                logger.warning("Layer 1 DB not found. Running without query refinement.")
                self.layer1_retriever = None

            logger.info("  - Initializing Google Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

            self.initialized = True
            logger.info("✅ RAG system initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise

    def download_pdf(self, url: str) -> str:
        """Download PDF from URL and save to temporary file."""
        try:
            logger.info(f"📥 Downloading PDF from: {url}")
            response = requests.get(str(url), timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb")
            tmp_file.write(response.content)
            tmp_file.close()
            
            logger.info(f"📄 PDF downloaded to: {tmp_file.name}")
            return tmp_file.name
            
        except Exception as e:
            logger.error(f"❌ Failed to download PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            logger.info(f"📖 Extracting text from PDF: {pdf_path}")
            reader = PyPDF2.PdfReader(pdf_path)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF.")
            
            logger.info(f"✅ Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract text: {e}")
            raise

    def answer_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Process PDF and answer multiple questions."""
        if not self.initialized:
            raise RuntimeError("RAG system is not initialized.")

        pdf_path = None
        try:
            # Download PDF
            pdf_path = self.download_pdf(pdf_url)
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            # Create document chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

            # Create vector store
            layer2_db = FAISS.from_documents(docs, self.embedding)
            layer2_retriever = layer2_db.as_retriever(search_kwargs={"k": 7})
            logger.info("  - In-memory vector store created.")

            def format_docs(docs: List[Document]) -> str:
                return "\n\n".join(doc.page_content for doc in docs)

            # Define prompts
            final_prompt = PromptTemplate.from_template(
                """You are an expert assistant. Answer the user's question based ONLY on the provided context.
If the answer is not in the context, say so. Do not make up information.

Context:
---
{context}
---
Question: {question}
Answer:"""
            )

            # Process each question
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"🤔 Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                refined_question = question
                if self.layer1_retriever:
                    logger.info("  - Refining question using Layer 1...")
                    refine_prompt = PromptTemplate.from_template(
                        """Refine the original question to be more specific using the initial context.

Original Question: {question}
Initial Context: {context}
---
Refined Question:"""
                    )
                    refine_query_chain = (
                        {"context": self.layer1_retriever | format_docs, "question": RunnablePassthrough()}
                        | refine_prompt
                        | self.llm
                        | StrOutputParser()
                    )
                    refined_question = refine_query_chain.invoke(question)
                    logger.info(f"  - Refined Question: {refined_question}")

                # Get final answer
                final_rag_chain = (
                    {
                        "context": layer2_retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | final_prompt
                    | self.llm
                    | StrOutputParser()
                )
                answer = final_rag_chain.invoke(refined_question)
                answers.append(answer)
                logger.info(f"✅ Answer {i+1} completed")

            return answers

        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")

        finally:
            # Clean up temporary file
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"🧹 Deleted temporary file: {pdf_path}")

# --- RAG System Instance ---
rag_system = RAGSystem()

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        logger.error("❌ GOOGLE_API_KEY environment variable not set.")
        return
    rag_system.initialize_system(gemini_api_key)

# --- Main Endpoint: /hackrx/run ---
@app.post("/hackrx/run", response_model=HackRxResponse, tags=["OST"])
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Process a PDF document from URL and answer multiple questions.
    Requires Bearer token authentication.
    """
    if not rag_system.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question is required.")

    try:
        logger.info(f"🚀 Processing {len(request.questions)} questions for document: {request.documents}")
        answers = rag_system.answer_questions(str(request.documents), request.questions)
        
        return HackRxResponse(answers=answers)

    except Exception as e:
        logger.error(f"❌ Failed to process request: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_initialized": rag_system.initialized,
        "version": "3.0.0"
    }