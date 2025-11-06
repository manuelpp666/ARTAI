# ================================================================
# train_tinyllama_kaggle.py — Fine-tuning TinyLlama-1.1B-Chat en dataset de arte (Kaggle GPU)
# ================================================================
# Instalar dependencias (solo la primera vez en una celda aparte):
# !pip install transformers==4.44.2 accelerate==0.33.0 datasets==2.21.0 tensorboard
# ================================================================

import os
import math
import gc
import re
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset

# ----------------------
# ⚙️ Configuración
# ----------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "/kaggle/working/models/tinyllama_arte_finetuned"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Usando dispositivo: {device}")

# Limpieza de memoria GPU
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ----------------------
# 📚 Cargar modelo y tokenizer
# ----------------------
print(f"📦 Cargando modelo base: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# 🧩 Cargar dataset desde Kaggle
# ----------------------
print("📂 Cargando dataset de texto desde Kaggle...")

# ✅ Carga directa desde el archivo .txt
dataset = load_dataset(
    "text",
    data_files={"train": "/kaggle/input/dataset-arte/dataset_completo.txt"}
)

# ✅ Limpieza y segmentación por secciones
def dividir_por_seccion(texto):
    # Divide el texto cada vez que encuentre una cabecera "SECCION ..."
    partes = re.split(r"SECCION\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑ\s]*", texto)
    secciones = [p.strip() for p in partes if len(p.strip()) > 50]
    return secciones

# Crear una lista con todas las secciones extraídas
textos_divididos = []
for t in dataset["train"]["text"]:
    textos_divididos.extend(dividir_por_seccion(t))

print(f"📄 Secciones extraídas: {len(textos_divididos)}")

# Convertir a dataset HuggingFace
dataset = Dataset.from_dict({"text": textos_divididos})

# Dividir en train/test
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ----------------------
# ✂️ Tokenización
# ----------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2,
    remove_columns=["text"]
)

print("✅ Tokenización completada.")

# ----------------------
# ⚖️ Data Collator
# ----------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------
# 📊 Métricas
# ----------------------
def compute_metrics(eval_pred):
    if isinstance(eval_pred, dict) and "eval_loss" in eval_pred:
        loss = eval_pred["eval_loss"]
    else:
        loss = None
    if loss is not None:
        perplexity = math.exp(loss) if loss < 10 else float("inf")
        return {"eval_loss": loss, "perplexity": perplexity}
    return {}

# ----------------------
# 🧠 Configuración de entrenamiento
# ----------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=200,
    fp16=torch.cuda.is_available(),
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=["tensorboard"],
    optim="adamw_torch",
)

# ----------------------
# 🚀 Entrenador HuggingFace
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ----------------------
# 🏋️ Entrenamiento
# ----------------------
print("🔥 Iniciando fine-tuning...")
trainer.train()

# ----------------------
# 💾 Guardar modelo final
# ----------------------
print("💾 Guardando modelo final...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# ----------------------
# 🧮 Evaluar modelo
# ----------------------
metrics = trainer.evaluate()
trainer.save_metrics("eval", metrics)
print("\n📈 Métricas finales:")
print(metrics)

# ----------------------
# 🖋️ Prueba de generación
# ----------------------
prompt = "SECCION Pablo Picasso SECCION"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.15,
)
print("\n💬 Texto generado:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
