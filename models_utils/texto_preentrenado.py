# ============================================================
# texto_preentrenado.py — Phi-3-mini + FAISS para arte (moderno)
# ============================================================

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# LangChain moderno
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 1️⃣ CONFIGURACIÓN DE RUTAS Y MODELOS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "arte_faiss_index")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

DEVICE = "cpu"  # Cambia a 'cuda' si tienes GPU

# ============================================================
# 2️⃣ CARGAR ÍNDICE FAISS
# ============================================================

print("🔁 Cargando índice FAISS local...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE}
)

vectorstore = FAISS.load_local(
    folder_path=INDEX_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("✅ Índice FAISS cargado correctamente.\n")

# ============================================================
# 3️⃣ CARGAR MODELO PHI-3-MINI (AUTOMÁTICO CPU / GPU)
# ============================================================

print("Cargando modelo Phi-3-mini (detección automática de hardware)...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_gpu = torch.cuda.is_available()

if use_gpu:
    try:
        from transformers import BitsAndBytesConfig
        print("💪 GPU detectada — activando cuantización 4-bit con bitsandbytes.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ No se pudo usar bitsandbytes: {e}\nCargando modelo normal en GPU.")
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
else:
    print("🧠 CPU detectada — cargando modelo en float32 (sin cuantización).")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

# Pipeline robusto (modo eager y sin DynamicCache)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32 if not use_gpu else torch.bfloat16,
    device_map="auto",
    temperature=0.6,
    max_new_tokens=512,
    model_kwargs={
        "use_cache": False,
        "attn_implementation": "eager"
    }
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

print("✅ Phi-3-mini listo (CPU/GPU compatible, modo eager activado)\n")

# ============================================================
# 4️⃣ PROMPT PERSONALIZADO
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Eres un experto en historia del arte. Responde en español con claridad y precisión, usando únicamente el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA (en español):
"""
)

# ============================================================
# 5️⃣ CONFIGURAR RETRIEVAL QA
# ============================================================

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ============================================================
# 6️⃣ FUNCIÓN DE CONSULTA
# ============================================================

def arte(pregunta):
    """Genera respuesta usando modelo preentrenado + FAISS"""
    print(f"\n{'='*70}")
    print(f"PREGUNTA: {pregunta}")
    print('='*70)

    result = qa.invoke({"query": pregunta})

    print(f"RESPUESTA: {result['result'].strip()}\n")

    print("FUENTES:")
    for d in result["source_documents"]:
        print(f" • {d.metadata.get('title', 'Sin título')}")

    return result['result']

# ============================================================
# 7️⃣ EJEMPLOS DE USO
# ============================================================

if __name__ == "__main__":
    arte("¿Qué es el arte?")
