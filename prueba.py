import os
import re

# Carpeta donde están tus archivos
input_folder = "datasets/español"
# Archivo final
output_file = "dataset_completo.txt"

# Función para limpiar el texto
def clean_text(text):
    # Eliminar encabezados y pies de Gutenberg
    text = re.sub(r'\*{3}.*?START OF.*?GUTENBERG.*?\*{3}', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'\*{3}.*?END OF.*?GUTENBERG.*?\*{3}', '', text, flags=re.IGNORECASE|re.DOTALL)
    # Solo conservar letras y espacios
    text = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñÜü\s]', ' ', text)
    # Reemplazar múltiples espacios por uno solo
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

all_texts = []

# Recorrer todos los archivos en la carpeta
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        path = os.path.join(input_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            cleaned = clean_text(text)
            all_texts.append(cleaned)

# Unir todo en un solo texto
final_text = "\n".join(all_texts)

# Guardar el archivo final
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"Archivo final creado: {output_file} ({len(final_text)} caracteres)")
