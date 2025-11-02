# ================================================================
# limpiar_wiki.py — Limpieza y preparación de dataset Wikipedia
# ================================================================
import os
import re

input_folder = "datasets/español/arte_traducido"
output_file = "dataset_wikipedia.txt"

# ---------------------------------------------------------------
# 🔹 Lista de abreviaturas que NO deben separar frases
# ---------------------------------------------------------------
abrevs = [
    "Sr", "Sra", "Dr", "Dra", "Lic", "Ing", "Prof",
    "a.C", "d.C",
    "etc", "pág", "Cap", "Ej", "No", "vs",
    "EE.UU", "U.S.A"
]

# ---------------------------------------------------------------
# 🔹 Función mejorada para separar frases
# ---------------------------------------------------------------
def separar_frases(text):
    # Patrón general: busca . ! ? seguido de espacio y mayúscula
    pattern = re.compile(r'([.!?])(\s+)(?=[A-ZÁÉÍÓÚÑÜ])')

    def reemplazo(match):
        start = match.start()
        # Tomamos hasta 10 chars antes del signo
        antes = text[max(0, start-10):start]

        # Revisar si termina con alguna abreviatura
        if any(antes.endswith(abrev) for abrev in abrevs):
            return match.group(1) + match.group(2)  # No cortar
        # Revisar si es número decimal
        if re.search(r'\d\.$', antes):
            return match.group(1) + match.group(2)
        # En otros casos, cortar línea
        return match.group(1) + '\n'

    return pattern.sub(reemplazo, text)


# ---------------------------------------------------------------
# 🔹 Función principal de limpieza
# ---------------------------------------------------------------
def clean_wiki_text(text):
    # --- 1. Quitar referencias y notas ---
    text = re.sub(r'<ref.*?>.*?</ref>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\[\d+\]', ' ', text)  # [1], [2]...
    text = re.sub(r'\[\[.*?\|.*?\]\]', lambda m: m.group(0).split('|')[1][:-2] if '|' in m.group(0) else '', text)
    text = re.sub(r'\[\[|\]\]', '', text)

    # --- 2. Convertir títulos de secciones en tokens especiales ---
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r' [SECCION] \1 [SECCION] ', text)

    # --- 3. Conservar letras, números y signos básicos ---
    text = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9.,;:!?()\'"¿¡\-\s—]', ' ', text)

    # --- 4. Eliminar repeticiones de signos ---
    text = re.sub(r'([.,;:!?()\'"¿¡\-—])\1+', r'\1', text)

    # --- 5. Reemplazar múltiples espacios por uno ---
    text = re.sub(r'\s+', ' ', text)

    # --- 6. Quitar espacios antes de puntuación ---
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # --- 7. Insertar saltos de línea útiles ---
    text = re.sub(r'\[SECCION\]', r'\n[SECCION]\n', text)
    text = separar_frases(text)

    # --- 8. Normalizar saltos de línea ---
    text = re.sub(r'\n+', '\n', text)

    return text.strip()

# ---------------------------------------------------------------
# 🔹 Procesar todos los archivos .txt
# ---------------------------------------------------------------
all_texts = []

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        path = os.path.join(input_folder, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            cleaned = clean_wiki_text(text)
            all_texts.append(cleaned)
            print(f"Procesado: {filename} ({len(cleaned)} caracteres limpios)")

# ---------------------------------------------------------------
# 🔹 Guardar resultado final
# ---------------------------------------------------------------
final_text = "\n".join(all_texts)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"\n✅ Archivo final creado: {output_file}")
print(f"Total caracteres: {len(final_text):,}")
