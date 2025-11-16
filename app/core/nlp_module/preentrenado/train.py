# ================================================================
# 
# Este script entrena un modelo de lenguaje basado en Transformer
# (fine-tuning de GPT-2 en español) — se usó un modelo preentrenado
# y se corrió en Kaggle.
# ================================================================

import re
import json
from pathlib import Path

def parse_wiki_txt(file_path):
    """
    Parsea el archivo .txt con formato:
    SECCION Título SECCION Contenido [FIN_SECCION]

    Comentarios:
    - Usa expresiones regulares para localizar secciones en el texto plano.
    - Las expresiones regulares permiten identificar patrones complejos
      en cadenas; aquí se usa re.findall para devolver todas las coincidencias.
    - El flag re.DOTALL hace que '.' también coincida con saltos de línea,
      permitiendo capturar contenido multilínea de cada sección.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Patrón corregido: captura título y contenido entre SECCION ... SECCION ... [FIN_SECCION]
    # Explicación del patrón:
    # - 'SECCION\\s+' busca la palabra literal 'SECCION' seguida de espacios.
    # - '(.*?)' captura de forma no codiciosa el título (grupo 1).
    # - '\\s+SECCION\\s+' localiza la siguiente palabra 'SECCION' que separa título y contenido.
    # - '(.*?)' captura el contenido (grupo 2), también no codicioso para que no coma más secciones.
    # - '\\s*\\[FIN_SECCION\\]' indica el final de la sección.
    # Este enfoque funciona bien si el archivo sigue consistentemente ese formato.
    pattern = r'SECCION\s+(.*?)\s+SECCION\s+(.*?)\s*\[FIN_SECCION\]'
    sections = re.findall(pattern, text, re.DOTALL)
    
    data = []
    for title, content in sections:
        # Limpieza del contenido:
        # - .strip() elimina espacios al inicio/fin
        # - re.sub(r'\s+', ' ', ...) normaliza múltiples espacios y saltos de línea a un solo espacio
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
# Esto evita fallos posteriores y da un mensaje claro si la ruta está mal.
if not Path(file_path).exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

# Parsea el dataset (usa la función basada en regex definida arriba)
data = parse_wiki_txt(file_path)

# Muestra las primeras 2 entradas como ejemplo — útil para comprobación rápida
for i, entry in enumerate(data[:2]):
    print(f"\n--- Entrada {i+1} ---")
    print(f"Título: {entry['title']}")
    print(f"Contenido (primeros 200 chars): {entry['content'][:200]}...")

# Opcional: Guarda como JSON para uso posterior (más fácil de cargar)
# Guardar en JSON es práctico porque luego Hugging Face o cualquier otro
# pipeline puede cargarlo fácilmente sin rehacer el parsing.
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
# Formateamos cada entrada como un bloque de texto (Título + Contenido) para LM causal.
texts = []
for item in data:
    # Formato: "Título: [título]\nContenido: [contenido]\n\n"
    # Esto le da al modelo un prompt natural y consistente para que aprenda estructura.
    text = f"Título: {item['title']}\nContenido: {item['content']}\n\n"
    texts.append(text)

# Crear Dataset de Hugging Face
# Dataset.from_dict crea un objeto Dataset optimizado para map, train_test_split, etc.
dataset = Dataset.from_dict({"text": texts})
print(f"Dataset creado con {len(dataset)} ejemplos.")


from transformers import AutoTokenizer, AutoModelForCausalLM

# Modelo preentrenado: GPT-2 en español (pequeño)
model_name = "datificate/gpt2-small-spanish"  # GPT-2 pequeño en español

# Tokenizer:
# - El tokenizer mapeará cadenas a IDs (enteros) que el modelo usa como entrada.
# - GPT-2 usa BPE (Byte Pair Encoding): un algoritmo de subpalabras que
#   une los pares de bytes/caracteres más frecuentes iterativamente, resultando
#   en tokens que son sub-palabras (útil para manejar vocabularios y palabras raras).
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Necesario si usas batching con padding

# Cargar el modelo causal para generación/autoregresiva
# AutoModelForCausalLM incorpora la arquitectura tipo GPT (solo decodificador).
model = AutoModelForCausalLM.from_pretrained(model_name)
# Ajusta la tabla de embeddings si el tokenizer fue modificado (por si se cambió vocab)
model.resize_token_embeddings(len(tokenizer))

print(f"Modelo cargado: {model_name}")


def tokenize_function(examples):
    """
    Tokeniza una lista de textos.
    - truncation=True: recorta textos más largos que max_length (evita overflow).
    - max_length=512: límite práctico por memoria GPU; ajustable según hardware.
    - padding=False: no pad aquí, lo gestiona el data_collator al hacer batches.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,        # Ajustado para GPU de Kaggle
        padding=False
    )

# Tokenizar el dataset: Dataset.map aplica tokenize_function por lotes (batched=True)
# Esto transforma cada ejemplo de texto en una estructura con 'input_ids' y 'attention_mask'.
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # removemos el texto crudo para ahorrar memoria
)

# Data collator para language modeling
# DataCollatorForLanguageModeling prepara los batches para LM causal.
# - Si mlm=False, se espera un LM causal (predecir la siguiente palabra).
# - Este collator puede crear los labels desplazados internamente para la loss.
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Dividir 95% train, 5% test
# train_test_split crea particiones reproducibles para entrenamiento y evaluación.
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05)
print(tokenized_dataset)


from transformers import Trainer, TrainingArguments

# Configuración de entrenamiento
# Comentarios sobre los parámetros clave:
# - num_train_epochs: cuántas pasadas sobre los datos.
# - per_device_train_batch_size: tamaño por GPU (bajo en Kaggle).
# - gradient_accumulation_steps: acumula gradientes para simular batches mayores
#   (útil cuando la GPU no permite batches grandes).
# - learning_rate: tasa de aprendizaje inicial (5e-5 es común en fine-tuning).
# - warmup_steps: pasos iniciales para incrementar gradualmente lr (mejora estabilidad).
# - eval_strategy="steps": evaluar cada cierto número de pasos.
# - fp16=True: usar media precisión (FP16) para ahorrar memoria y acelerar.
# - load_best_model_at_end=True: al final recupera el modelo con mejor eval_loss.
# - metric_for_best_model="eval_loss": usamos pérdida de evaluación como criterio.
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
    # En versiones antiguas el argumento era evaluation_strategy; según versión puede variar.
    eval_strategy="steps",      
    eval_steps=500,
    
    fp16=True,
    dataloader_num_workers=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer: encapsula loop de entrenamiento, evaluación, scheduler, optimizador, checkpoints.
# Implementa internamente:
# - Optimizer (AdamW): Adam con decaimiento de pesos (weight decay).
# - Scheduler de learning-rate: maneja warmup y decaimiento posterior.
# - Cálculo de loss: Cross-Entropy entre las probabilidades predichas y los tokens reales.
# - Backpropagation y actualización de parámetros.
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

# Guardar modelo final y tokenizer:
# - save_pretrained guarda pesos y configuración en formato compatible con Transformers.
final_output_dir = "/kaggle/working/art_gpt_arte_final"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print(f"Modelo guardado en: {final_output_dir}")
