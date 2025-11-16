# ============================================================
# CELDA 1 - IMPORTS Y CONFIGURACIÓN INICIAL
# ============================================================

# Carga utilidades de LangChain para manejar documentos,
# dividir textos en fragmentos (chunks), generar embeddings,
# indexarlos en FAISS y crear pipelines de Q&A.
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Transformes para cargar el modelo generativo Phi-3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

import torch
import warnings
warnings.filterwarnings("ignore")  # Ignorar warnings del tokenizer

import re
from langchain.schema import Document


# ============================================================
# CARGA DEL DATASET RAW Y CONVERSIÓN A DOCUMENTOS
# ============================================================

ruta = "/kaggle/input/dataset-arte/dataset_completo.txt"

# Leer el archivo completo
with open(ruta, 'r', encoding='utf-8') as f:
    texto = f.read()

# Dividir el dataset en secciones usando el delimitador [FIN_SECCION]
secciones = re.split(r'\[FIN_SECCION\]', texto)
documents = []

# Procesar cada sección para extraer título y contenido
for sec in secciones:
    sec = sec.strip()
    if len(sec) < 50:
        continue  # Ignorar secciones muy cortas

    # Buscar patrón: "SECCION <TÍTULO> SECCION"
    match = re.match(
        r'SECCION\s+([A-Za-zÁÉÍÓÚáéíóúñÑ0-9\s\-\.,]+?)\s+SECCION', 
        sec
    )

    if match:
        # Si encuentra un título correcto
        titulo = match.group(1).strip()
        contenido = sec[match.end():].strip()
    else:
        # Fallback: detectar al menos un "SECCION <TÍTULO>"
        match2 = re.match(r'SECCION\s+(.+)', sec)
        if not match2:
            continue
        titulo = match2.group(1).split(' ')[0].strip()
        contenido = sec[match2.end():].strip()

    # Crear un documento LangChain con metadatos
    documents.append(Document(
        page_content=contenido,
        metadata={"title": titulo, "tipo": "arte"}
    ))

print(f"🎨 Dataset cargado: {len(documents)} artistas/movimientos")
if documents:
    print(f"📄 Ejemplo: {documents[0].metadata['title']}")
    print(f"📝 Fragmento: {documents[0].page_content[:200]}...")


# ============================================================
# CELDA 2 - CREACIÓN O CARGA DEL ÍNDICE FAISS
# ============================================================

from tqdm import tqdm
import os, time

index_path = "arte_faiss_index"  # Carpeta donde se guarda el índice FAISS

# Cargar modelo de embeddings multilingüe de Sentence Transformers
print("Cargando modelo de embeddings...")
start = time.time()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}  # Ejecutar en CPU
)

print(f"Embeddings listos en {time.time() - start:.1f}s\n")


# ============================================================
# SI YA EXISTE UN ÍNDICE FAISS → CARGAR
# ============================================================
if os.path.exists(index_path):
    print("🔁 Cargando índice FAISS guardado...")

    # Cargar FAISS desde disco
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Crear un retriever para buscar los k=4 documentos más similares
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("✅ Índice FAISS cargado correctamente.\n")

# ============================================================
# SI NO EXISTE → CREAR ÍNDICE Y GUARDARLO
# ============================================================
else:
    print("⚙️ No se encontró un índice guardado, creando desde cero...")
    start = time.time()

    # === CHUNKING ===
    # Dividir los documentos en fragmentos pequeños
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=70,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"Total de chunks: {len(chunks)}")

    # === GENERAR EMBEDDINGS E INDEXAR FAISS ===
    print("Calculando embeddings e indexando FAISS (2.5-4 min)...")
    text_embedding_pairs = []

    # Generar embeddings para cada chunk con barra de progreso
    for chunk in tqdm(chunks, desc="Embeddings", unit="chunk"):
        embedding = embeddings.embed_query(chunk.page_content)
        text_embedding_pairs.append((chunk.page_content, embedding))

    # Crear el índice FAISS
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in chunks]
    )

    # Guardar el índice
    vectorstore.save_local(index_path)
    print(f"💾 Índice FAISS guardado en '{index_path}/'")

    # Crear retriever (k=4 resultados más relevantes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    total_time = time.time() - start
    print(f"✅ BASE DE ARTE INDEXADA EN {total_time:.1f}s ({total_time/60:.1f} min)")


# ============================================================
# CELDA 3 - CONFIGURACIÓN DEL MODELO GENERATIVO Phi-3-mini
# ============================================================

model_id = "microsoft/phi-3-mini-4k-instruct"
print("Cargando modelo Phi-3-mini-4k-instruct (4-bit)...")

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Asegurar que el modelo tenga un token de padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuración de cuantización 4-bit (optimización de VRAM)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Cargar modelo Phi-3-mini cuantizado
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Crear pipeline de generación con fixes para DynamicCache
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    temperature=0.6,      # Control de creatividad
    max_new_tokens=512,   # Longitud máxima de respuesta
    model_kwargs={
        "use_cache": False,            # Evitar errores de cache dinámica
        "attn_implementation": "eager" # Atención estable
    }
)

# Convertir pipeline en LLM compatible con LangChain
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("✅ Phi-3-mini listo (DynamicCache desactivado y modo eager activado)")


# ============================================================
# CELDA 4 - DEFINICIÓN DEL PROMPT PERSONALIZADO (INSTRUCCIONES)
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Eres un experto en historia del arte. 
Responde en español con claridad y precisión, usando únicamente el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA (en español):"""
)


# ============================================================
# CELDA 5 - CREAR EL CHATBOT DE ARTE (RAG: Retrieval + LLM)
# ============================================================

# Construir el sistema Q&A de LangChain que integra:
# - recuperador FAISS (contexto)
# - modelo Phi-3 (generación)
# - prompt (formato de respuesta)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Inserta todo el contexto en el prompt
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Función interfaz del chatbot
def arte(pregunta):
    print(f"\n{'='*70}")
    print(f"PREGUNTA: {pregunta}")  # Mostrar pregunta original
    print('='*70)

    # Ejecutar la cadena Q&A
    result = qa.invoke({"query": pregunta})

    # Imprimir la respuesta generada
    print(f"RESPUESTA: {result['result'].strip()}\n")

    # Mostrar las fuentes usadas
    print("FUENTES:")
    for d in result["source_documents"]:
        print(f"  • {d.metadata.get('title', 'Sin título')}")


# ============================================================
# CELDA 6 - PRUEBAS DEL CHATBOT
# ============================================================

arte("¿Qué es el arte?")
arte("¿Quién fue Robert Delaunay?")
arte("¿Quién es Vicent Van Gogh?")
arte("¿Qué es el simultaneísmo?")
arte("Obras famosas de Delaunay sobre ritmo")
