import os
from preprocess import construir_vocab
import json

# ----------------------
# Rutas
# ----------------------
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/arte_traducido/dataset_completo1.txt")
ruta_vocab = "bpe_tokenizer.json"

# ----------------------
# Construir / cargar tokenizer
# ----------------------
print("🔹 Construyendo tokenizer...")
tokenizer, stoi, itos = construir_vocab(ruta_dataset, ruta_vocab=ruta_vocab, vocab_size=10000)

# ----------------------
# Chequeo tokens especiales
# ----------------------
tokens_especiales = ["SECCION", "[FIN_SECCION]"]
for t in tokens_especiales:
    token_ids = tokenizer.encode(t).ids
    print(f"Token '{t}' codificado como IDs: {token_ids}")
    if len(token_ids) != 1:
        print(f"⚠️ ATENCIÓN: '{t}' no es un token único. Esto puede causar errores de generación.")

# ----------------------
# Chequeo del dataset
# ----------------------
print("\n🔹 Analizando dataset...")
with open(ruta_dataset, "r", encoding="utf-8") as f:
    lineas = f.readlines()

longitudes = [len(line.split()) for line in lineas]
print(f"Total de líneas: {len(lineas)}")
print(f"Longitud mínima de línea: {min(longitudes)} palabras")
print(f"Longitud máxima de línea: {max(longitudes)} palabras")
print(f"Longitud promedio de línea: {sum(longitudes)/len(longitudes):.2f} palabras")

# Mostrar ejemplos de líneas largas (>500 palabras)
print("\n🔹 Ejemplos de líneas largas (>500 palabras):")
for i, l in enumerate(lineas):
    if len(l.split()) > 500:
        print(f"- Línea {i}: {len(l.split())} palabras")
        print("  ", l[:200], "...\n")  # Muestra primeros 200 caracteres

# ----------------------
# Chequeo de delimitador [FIN_SECCION]
# ----------------------
count_fin_seccion = sum(1 for l in lineas if "[FIN_SECCION]" in l)
print(f"\nSecciones con [FIN_SECCION]: {count_fin_seccion} de {len(lineas)} líneas")
if count_fin_seccion != len(lineas):
    print("⚠️ Algunas líneas no contienen '[FIN_SECCION]'. Esto puede romper la generación autoregresiva.")
