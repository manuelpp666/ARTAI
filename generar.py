# ============================================================
# generar_index_local.py — Crear índice FAISS + pickle
# ============================================================

import os
import re
import time
import pickle
from tqdm import tqdm

# 🚨 Actualización de imports para LangChain 1.0.5 moderno
# Documentos
from langchain_community.docstore.document import Document

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS
from langchain_community.vectorstores import FAISS


# ============================================================
# 1️⃣ CONFIGURACIÓN
# ============================================================

DATASET_PATH = "dataset_completo.txt"  # Cambia a tu ruta local
INDEX_PATH = "./arte_faiss_index"
PICKLE_PATH = "./arte_index.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cpu"  # "cuda" si tienes GPU

# ============================================================
# 2️⃣ CARGAR DOCUMENTOS
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en: {DATASET_PATH}")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    texto = f.read()

# Dividir por el delimitador final
secciones = re.split(r'\[FIN_SECCION\]', texto)
documents = []

for sec in secciones:
    sec = sec.strip()
    if len(sec) < 50:
        continue

    # Detecta el primer título: "SECCION <nombre>"
    match = re.match(r'SECCION\s+([A-Za-zÁÉÍÓÚáéíóúñÑ0-9\s\-\.,]+?)\s+SECCION', sec)
    if match:
        titulo = match.group(1).strip()
        contenido = sec[match.end():].strip()
    else:
        # fallback: si no tiene dos SECCION, intenta al menos el primero
        match2 = re.match(r'SECCION\s+(.+)', sec)
        if not match2:
            continue
        titulo = match2.group(1).split(' ')[0].strip()
        contenido = sec[match2.end():].strip()

    documents.append(Document(
        page_content=contenido,
        metadata={"title": titulo, "tipo": "arte"}
    ))

print(f"🎨 Documentos cargados: {len(documents)}")
if documents:
    print(f"📄 Ejemplo: {documents[0].metadata['title']}")
    print(f"📝 Fragmento: {documents[0].page_content[:200]}...\n")

# ============================================================
# 3️⃣ CARGAR EMBEDDINGS
# ============================================================

print("🔁 Cargando modelo de embeddings...")
start = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE}
)
print(f"✅ Embeddings listos en {time.time() - start:.1f}s\n")

# ============================================================
# 4️⃣ CREAR ÍNDICE FAISS + PICKLE
# ============================================================

print("⚙️ Creando índice FAISS desde cero...")
start = time.time()

# === CHUNKING ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=70,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
print(f"Total de chunks: {len(chunks)}")

# === EMBEDDINGS + INDEXACIÓN ===
text_embedding_pairs = []
for chunk in tqdm(chunks, desc="Embeddings", unit="chunk"):
    embedding = embeddings.embed_query(chunk.page_content)
    text_embedding_pairs.append((chunk.page_content, embedding))

vectorstore = FAISS.from_embeddings(
    text_embeddings=text_embedding_pairs,
    embedding=embeddings,
    metadatas=[chunk.metadata for chunk in chunks]
)

# Guardar FAISS
os.makedirs(INDEX_PATH, exist_ok=True)
vectorstore.save_local(INDEX_PATH)

# Guardar pickle
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(vectorstore, f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

total_time = time.time() - start
print(f"✅ Índice FAISS y pickle creados en {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"💾 FAISS: {INDEX_PATH}/")
print(f"💾 Pickle: {PICKLE_PATH}")
