import re
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
==========================================
CONFIGURATION
==========================================
PDF_PATH = "troubleshooting_runbook.pdf"
COLLECTION_NAME = "contoso_runbook_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

os.environ["GEMINI_API_KEY"]=getpass.getpass("Enter Gemini API Key")

client = genai.Client()

==========================================
PART 1: DATA UNDERSTANDING & PREPROCESSING
==========================================
def load_and_explore_pdf(pdf_path: str):
"""Loads PDF using PyPDFLoader, inspects structure, and cleans extracted text."""
loader = PyPDFLoader(pdf_path)
documents = loader.load()
num_pages = len(documents)

print("--- Part 1: Document Exploration ---")
print(f"File: {pdf_path} | Total Pages: {num_pages}")

for doc in documents:
    page_num = doc.metadata.get("page", 0) + 1
    doc.metadata["page"] = page_num
    
    cleaned_text = re.sub(
      r"Contoso Platform Engineering | Troubleshooting Runbook v3.2 | Internal - Synthetic demo data\sPage \d+",
      "",
      doc.page_content,
    )
    doc.page_content = re.sub(r"
      {3,}", "
      ", cleaned_text).strip()
      print(f"Page {page_num}: Extracted {len(doc.page_content)} characters.")
return documents

def chunk_documents(documents: list, chunk_size: int = 800, chunk_overlap: int = 120):
    """Splits LangChain Documents into overlapping chunks while preserving metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=["
      ", "
      ", ". ", " ", ""],
    )
  split_docs = text_splitter.split_documents(documents)
  print(f"Total chunks created: {len(split_docs)}")
  return split_docs

==========================================
PART 2: EMBEDDING & VECTOR DATABASE
==========================================
def build_vector_store(split_docs: list):
  """Generates dense embeddings and indexes LangChain Document chunks in ChromaDB."""
  print("--- Part 2: Building Vector Database ---")
  chroma_client = chromadb.Client()
  sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
  model_name=EMBEDDING_MODEL_NAME
  )
  try:
  chroma_client.delete_collection(name=COLLECTION_NAME)
  except Exception:
  pass
  collection = chroma_client.create_collection(
  name=COLLECTION_NAME,
  embedding_function=sentence_transformer_ef,
  metadata={"hnsw:space": "cosine"},
  )
  collection.add(
  documents=[doc.page_content for doc in split_docs],
  metadatas=[doc.metadata for doc in split_docs],
  ids=[f"chunk_{i}" for i in range(len(split_docs))],
  )
  print(f"Indexed {collection.count()} chunks into ChromaDB using '{EMBEDDING_MODEL_NAME}'.")
  return collection

==========================================
PART 3: RETRIEVAL PIPELINE
==========================================
def retrieve_top_k(collection, query: str, k: int = 3):
    """Embeds the user query and retrieves top-k semantically similar chunks."""
    results = collection.query(
    query_texts=[query],
    n_results=k,
    include=["documents", "metadatas", "distances"],
    )
    retrieved_items = []
    for i in range(len(results["documents"][0])):
    retrieved_items.append({
    "rank": i + 1,
    "text": results["documents"][0][i],
    "page": results["metadatas"][0][i]["page"],
    "similarity_score": round(1 - results["distances"][0][i], 4),
    })
    return retrieved_items

==========================================
PART 4: ANSWER GENERATION (RAG)
==========================================
def generate_grounded_answer(query: str, retrieved_chunks: list) -> str:
    """Generates an answer strictly grounded in the retrieved context."""
    context_block = "
    ________________________________________
    ".join(
    [f"[Page {item['page']}]
    {item['text']}" for item in retrieved_chunks]
    )
    system_instruction = (
      "You are a strict, technical Knowledge Assistant. Answer the user's question "
      "USING ONLY the facts explicitly provided in the Context below.
      "
      "Rules:
      "
      "1. Do not use external knowledge, assumptions, or extrapolation.
      "
      "2. Cite the page number (e.g., [Page X]) for every claim you make.
      "
      "3. If the answer cannot be found in the provided context, respond strictly with: "
      "'I cannot answer this based on the provided document.'"
    )
    user_prompt = f"""Context:
    {context_block}
    Question: {query}
    Grounded Answer:"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        ),
    )
return response.text

==========================================
PART 5: END-TO-END RAG APPLICATION
==========================================
def main():
    # 1. Load & Preprocess via PyPDFLoader
    documents = load_and_explore_pdf(PDF_PATH)
    split_docs = chunk_documents(documents, chunk_size=800, chunk_overlap=120)
    
    # 2. Index in Vector DB
    vector_db = build_vector_store(split_docs)
    
    # 3. Experiment with Top-k values
    sample_query = "How do I fix ERR-401?"
    print(f"--- Part 3: Top-k Retrieval Experiment (Query: '{sample_query}') ---")
    for k_val in [3, 5, 7]:
      hits = retrieve_top_k(vector_db, sample_query, k=k_val)
      pages_hit = [h["page"] for h in hits]
      top_score = hits[0]["similarity_score"]
      print(f"k={k_val} | Pages Retrieved: {pages_hit} | Top Cosine Similarity: {top_score}")
    print()
    
    # 4. Run Multiple Queries
    test_queries = [
      "What are the golden rules for on-call engineers?",
      "How do I diagnose and resolve DB-POOL connection exhaustion?",
      "What happened during the week of 10 August 2026?",
      "What is the transformer self-attention formula?"
    ]
    
    print("--- Part 4 & 5: End-to-End Grounded Q&A Demonstration ---")
    for q in test_queries:
      print(f"QUERY: {q}")
      retrieved = retrieve_top_k(vector_db, q, k=3)
      answer = generate_grounded_answer(q, retrieved)
      print(f"ANSWER: {answer}")
      print("-" * 60)

if __name__ == "__main__":
main()
