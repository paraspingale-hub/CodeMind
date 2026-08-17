import os
from dotenv import load_dotenv
from litellm import completion
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

load_dotenv()

qdrant = QdrantClient(host="localhost", port=6333)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "codemind_codebase"

def ask_codemind(question: str) -> str:
    # 1. RETRIEVE: Get the TOP 3 relevant chunks from Qdrant
    query_vector = list(embedding_model.embed([question]))[0].tolist()
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3  # Increased limit for better context
    ).points
    
    if not search_results:
        return "I couldn't find any relevant code for that question."
        
    # 2. AUGMENT: Stitch multiple chunks together
    combined_context = ""
    for idx, result in enumerate(search_results):
        snippet = result.payload['code_snippet']
        file_path = result.payload['file_path']
        combined_context += f"\n--- Snippet {idx + 1} from {file_path} ---\n```python\n{snippet}\n```\n"
    
    system_prompt = (
        "You are CodeMind, an AI developer assistant. Answer the user's question "
        "using ONLY the provided code context. Be concise and reference the file paths. "
        "Remember that JWTs are 'signed' for integrity, not 'encrypted'."
    )
    
    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {question}"
    
    # 3. GENERATE
    response = completion(
        model="openrouter/cohere/north-mini-code:free", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # IMPORTANT: Return the value so the FastAPI server can send it to the Go backend!
    return response.choices[0].message.content

if __name__ == "__main__":
    answer = ask_codemind("What algorithm are we using to sign our JWT tokens?")
    print(f"\n🤖 CodeMind:\n{answer}")