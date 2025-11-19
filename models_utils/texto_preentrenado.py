# ============================================================
# texto_preentrenado.py — Implementación Manual (Sin Chains)
# ============================================================

import os
import torch
from transformers import AutoTokenizer, pipeline

# Usamos solo los componentes base que sí te funcionan
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 1️⃣ CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ajustamos la ruta para subir dos niveles: models_utils -> ARTAI -> raiz
INDEX_PATH = os.path.join(BASE_DIR,"..", "arte_faiss_index")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Dispositivo: {DEVICE.upper()}")

# ============================================================
# 2️⃣ CARGAR MODELOS
# ============================================================

# A. Embeddings
print("1. Cargando Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE}
)

# B. Vector Store (FAISS)
print("2. Cargando Índice FAISS...")
try:
    vectorstore = FAISS.load_local(
        folder_path=INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ FAISS cargado.")
except Exception as e:
    print(f"⚠️ Error cargando FAISS: {e}")
    retriever = None

# C. Modelo de Lenguaje (Phi-3)
print("3. Cargando LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

model_kwargs = {
    "device_map": "auto",
    "trust_remote_code": True,
    "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32
}

text_generation_pipeline = pipeline(
    "text-generation",
    model=LLM_MODEL_ID,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,
    do_sample=True,
    return_full_text=False, # Importante para no repetir el prompt
    **model_kwargs
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("✅ LLM listo.")

# ============================================================
# 3️⃣ CLASE QA MANUAL (Reemplaza a RetrievalQA)
# ============================================================

class ManualQA:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def invoke(self, inputs):
        """
        Simula el comportamiento de qa.invoke({'query': '...'}) con limpieza de texto.
        """
        query = inputs.get("query", "")
        if not query:
            return {"result": "Error: Pregunta vacía."}
            
        if not self.retriever:
            return {"result": "Error: El sistema de memoria (FAISS) no se pudo cargar."}

        # 1. RECUPERACIÓN
        try:
            docs = self.retriever.invoke(query)
        except:
            docs = self.retriever.get_relevant_documents(query)
        
        # 2. CONTEXTO
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 3. PROMPT
        prompt = f"""<|system|>
Eres un experto en historia del arte. Responde a la pregunta basándote únicamente en el siguiente contexto proporcionado.
Si la respuesta no está en el contexto, di que no lo sabes.

CONTEXTO:
{context_text}
<|end|>
<|user|>
{query}
<|end|>
<|assistant|>"""

        # 4. GENERACIÓN
        print(f"🔎 Consultando LLM con contexto de {len(docs)} documentos...")
        raw_response = self.llm.invoke(prompt)
        
        # --- NUEVO: LIMPIEZA DE RESPUESTA ---
        # Si la respuesta no termina en puntuación, cortamos hasta el último punto.
        respuesta_final = raw_response.strip()
        if respuesta_final and respuesta_final[-1] not in [".", "!", "?"]:
            # Buscamos el último punto final
            ultimo_punto = respuesta_final.rfind(".")
            if ultimo_punto > 0:
                respuesta_final = respuesta_final[:ultimo_punto+1]
            else:
                # Si no hay puntos, dejamos la respuesta tal cual (o podrías añadir "...")
                pass
        # ------------------------------------
        
        # 5. RETORNO
        return {
            "result": respuesta_final,
            "source_documents": docs
        }

# ============================================================
# 4️⃣ INSTANCIA GLOBAL
# ============================================================

# Esta es la variable 'qa' que importa tu main.py
qa = ManualQA(llm, retriever)

if __name__ == "__main__":
    # Prueba rápida si ejecutas este archivo directo
    resp = qa.invoke({"query": "¿Qué es el impresionismo?"})
    print("\nRESPUESTA FINAL:", resp["result"])