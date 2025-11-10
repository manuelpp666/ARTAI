# ================================================================
# 
# Este script entrena un modelo de lenguaje basado en Transformer
# se usó un modelo preentrenado y se corrio en Kaggle 
# ================================================================


import re
import json
from pathlib import Path

def parse_wiki_txt(file_path):
    """
    Parsea el archivo .txt con formato:
    SECCION Título SECCION Contenido [FIN_SECCION]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Patrón corregido: captura título y contenido entre SECCION ... SECCION ... [FIN_SECCION]
    pattern = r'SECCION\s+(.*?)\s+SECCION\s+(.*?)\s*\[FIN_SECCION\]'
    sections = re.findall(pattern, text, re.DOTALL)
    
    data = []
    for title, content in sections:
        # Limpieza del contenido
        clean_content = content.strip()
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Normaliza espacios
        clean_title = title.strip()
        
        data.append({
            'title': clean_title,
            'content': clean_content
        })
    
    print(f"Parseadas {len(data)} secciones correctamente.")
    return data

# === EJECUCIÓN EN KAGGLE ===
file_path = '/kaggle/input/dataset-arte/dataset_completo.txt'

# Verifica que el archivo existe
if not Path(file_path).exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

# Parsea el dataset
data = parse_wiki_txt(file_path)

# Muestra las primeras 2 entradas como ejemplo
for i, entry in enumerate(data[:2]):
    print(f"\n--- Entrada {i+1} ---")
    print(f"Título: {entry['title']}")
    print(f"Contenido (primeros 200 chars): {entry['content'][:200]}...")

# Opcional: Guarda como JSON para uso posterior (más fácil de cargar)
output_path = '/kaggle/working/arte_dataset_parsed.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nDataset parseado y guardado en: {output_path}")



import json
from datasets import Dataset

# Cargar el JSON que ya generaste
json_path = '/kaggle/working/arte_dataset_parsed.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total de artículos cargados: {len(data)}")
print(f"Ejemplo: {data[0]['title']}")



# Convertir a formato de texto para language modeling
texts = []
for item in data:
    # Formato: "Título: [título]\nContenido: [contenido]\n\n"
    text = f"Título: {item['title']}\nContenido: {item['content']}\n\n"
    texts.append(text)

# Crear Dataset de Hugging Face
dataset = Dataset.from_dict({"text": texts})
print(f"Dataset creado con {len(dataset)} ejemplos.")


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "datificate/gpt2-small-spanish"  # GPT-2 pequeño en español

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Necesario

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

print(f"Modelo cargado: {model_name}")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,        # Ajustado para GPU de Kaggle
        padding=False
    )

# Tokenizar
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Data collator para language modeling
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Dividir 95% train, 5% test
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05)
print(tokenized_dataset)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/kaggle/working/art_gpt_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    
    # CORREGIDO: 'evaluation_strategy' → 'eval_strategy'
    eval_strategy="steps",      
    eval_steps=500,
    
    fp16=True,
    dataloader_num_workers=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Iniciando entrenamiento...")
trainer.train()
print("Entrenamiento completado!")

# Guardar modelo final
final_output_dir = "/kaggle/working/art_gpt_arte_final"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print(f"Modelo guardado en: {final_output_dir}")