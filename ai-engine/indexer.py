import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

COLLECTION_NAME = "codemind_codebase"

# 1. Initialize Clients
client = QdrantClient(host="localhost", port=6333)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Ensure Collection Exists
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✅ Created Qdrant collection: '{COLLECTION_NAME}'")

# 3. Setup the Python-aware splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=300, 
    chunk_overlap=50
)

def generate_stable_id(file_path: str, chunk_index: int) -> str:
    """Generates a stable UUID based on the file path and chunk index."""
    namespace = uuid.NAMESPACE_URL
    return str(uuid.uuid5(namespace, f"{file_path}_{chunk_index}"))

def index_directory(directory_path: str):
    points = []
    
    print(f"📂 Scanning directory: {directory_path}...")
    
    # Walk through all files in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if not file.endswith(".py"):
                continue  # Skip non-Python files for now
                
            file_path = os.path.join(root, file)
            print(f"   Reading {file_path}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_code = f.read()
                
            # Split code into semantic chunks
            code_chunks = python_splitter.split_text(raw_code)
            embeddings = list(embedding_model.embed(code_chunks))
            
            # Package into Qdrant PointStructs with stable UUIDs
            for idx, (chunk, vector) in enumerate(zip(code_chunks, embeddings)):
                points.append(
                    PointStruct(
                        id=generate_stable_id(file_path, idx),
                        vector=vector.tolist(),
                        payload={
                            "file_path": file_path,
                            "code_snippet": chunk,
                            "language": "python"
                        }
                    )
                )

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"🚀 Successfully indexed {len(points)} total points into Qdrant!")
    else:
        print("⚠️ No Python files found to index.")

if __name__ == "__main__":
    # Now it indexes the entire sample_code folder, not just one file
    index_directory("sample_code")