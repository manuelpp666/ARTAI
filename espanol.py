import os
import re
from PyPDF2 import PdfReader

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
input_folder = "datasets/español/arte_espanol"
output_folder = "datasets/español/arte_traducido"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def limpiar_texto(texto: str) -> str:
    """Limpia texto: elimina saltos, espacios y números sueltos"""
    texto = texto.replace("\n", " ")
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\b\d+\b', '', texto)  # quita números de página o índices
    return texto.strip()

# -----------------------------
# CONVERSIÓN PDF → TXT
# -----------------------------
for filename in os.listdir(input_folder):
    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

    print(f"\n📘 Procesando: {filename}")

    # Saltar si ya fue convertido
    if os.path.exists(output_path):
        print(f"⚠️ Ya existe {output_path}, saltando...")
        continue

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"📄 Total de páginas: {num_pages}")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()

            # Ignorar páginas vacías o solo con imágenes
            if not page_text or len(page_text.strip()) < 20:
                continue

            texto_limpio = limpiar_texto(page_text)
            f_out.write(texto_limpio + "\n")
            print(f"   ✅ Página {i+1}/{num_pages} guardada.")

    print(f"🎨 Conversión terminada y guardada en: {output_path}")

print("\n✅ Todos los PDFs fueron convertidos correctamente.")
