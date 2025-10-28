import os
import re

# Carpeta donde están tus archivos
input_folder = "datasets/español/arte_traducido"
# Archivo final
output_file = "dataset_completo.txt"

def clean_text(text):
    # --- 1. Eliminar encabezados/pies de Gutenberg u otros metadatos ---
    text = re.sub(r'\*{3}.*?(START|BEGINNING) OF.*?GUTENBERG.*?\*{3}', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\*{3}.*?(END|FINISHED) OF.*?GUTENBERG.*?\*{3}', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # --- 2. Quitar posibles URLs, correos o patrones extraños ---
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # --- 3. Conservar letras, números y signos de puntuación básica ---
    # Permitimos letras (mayúsculas/minúsculas), acentos, ñ, números y .,;:?!() comillas
    text = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9.,;:!?()\'"¿¡\-\s]', ' ', text)

    # --- 4. Eliminar repeticiones de signos (ej: "??!!", "....") ---
    text = re.sub(r'([.,;:!?()\'"¿¡\-])\1+', r'\1', text)
    
    # --- 5. Reemplazar múltiples espacios o saltos de línea por uno ---
    text = re.sub(r'\s+', ' ', text)
    
    # --- 6. Quitar espacios antes de puntuación ---
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    return text.strip()


all_texts = []

# Recorrer todos los archivos en la carpeta
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        path = os.path.join(input_folder, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            cleaned = clean_text(text)
            all_texts.append(cleaned)
            print(f"Procesado: {filename} ({len(cleaned)} caracteres limpios)")

# Unir todo en un solo texto
final_text = "\n".join(all_texts)

# Guardar el archivo final
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"\n✅ Archivo final creado: {output_file}")
print(f"Total caracteres: {len(final_text):,}")
