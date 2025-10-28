import os
import time
import re
import pdfplumber
from deep_translator import GoogleTranslator

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
input_folder = "datasets/español/arte"
output_folder = "datasets/español/arte_traducido"
os.makedirs(output_folder, exist_ok=True)

MAX_CHARS = 4800   # límite seguro para evitar error de 5000 caracteres
PAUSA_SEGUNDOS = 2 # espera entre fragmentos para no saturar el traductor

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def limpiar_texto(texto: str) -> str:
    """Limpia texto: elimina saltos, espacios, números sueltos"""
    texto = texto.replace("\n", " ")
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\b\d+\b', '', texto)  # quita números de página o índices
    return texto.strip()

def dividir_texto(texto: str, max_len=MAX_CHARS):
    """Divide texto en fragmentos manejables para el traductor"""
    return [texto[i:i + max_len] for i in range(0, len(texto), max_len)]

def extraer_texto_por_columnas(page):
    """
    Detecta si hay columnas y devuelve el texto ordenado correctamente.
    Si es una columna, devuelve todo como una sola cadena.
    """
    width = page.width
    text_full = page.extract_text()  # extrae todo
    if not text_full or len(text_full.strip()) < 20:
        return ""  # página vacía

    # heurística simple: si la longitud de líneas y saltos sugiere columnas
    lineas = [l for l in text_full.split("\n") if l.strip()]
    if len(lineas) < 3:  # pocas líneas, probablemente no hay columnas
        return text_full

    # Intentar detectar columnas: extraer izquierda y derecha
    left_col = page.within_bbox((0, 0, width/2, page.height)).extract_text()
    right_col = page.within_bbox((width/2, 0, width, page.height)).extract_text()

    # Si ambos tienen texto y son significativamente largos → columnas
    if left_col and right_col and len(left_col.strip()) > 20 and len(right_col.strip()) > 20:
        return (left_col or "") + "\n" + (right_col or "")
    else:
        return text_full  # página de una sola columna

# -----------------------------
# TRADUCCIÓN DE PDFS
# -----------------------------
translator = GoogleTranslator(source='en', target='es')

for filename in os.listdir(input_folder):
    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

    print(f"\n📘 Procesando: {filename}")

    if os.path.exists(output_path):
        print(f"⚠️ Ya existe {output_path}, saltando...")
        continue

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        print(f"📄 Total de páginas: {num_pages}")

        with open(output_path, "a", encoding="utf-8") as f_out:
            for i, page in enumerate(pdf.pages):
                page_text = extraer_texto_por_columnas(page)
                if not page_text:
                    continue

                texto_limpio = limpiar_texto(page_text)
                partes = dividir_texto(texto_limpio)

                for j, parte in enumerate(partes):
                    print(f"   🔤 Traduciendo pág {i+1}/{num_pages}, frag {j+1}/{len(partes)}...")
                    try:
                        traducido = translator.translate(parte)
                        f_out.write(traducido + "\n")
                        print(f"      ✅ Guardado fragmento ({i+1}.{j+1})")
                    except Exception as e:
                        print(f"      ⚠️ Error en pág {i+1}, frag {j+1}: {e}")
                        time.sleep(10)
                    time.sleep(PAUSA_SEGUNDOS)

    print(f"🎨 Traducción terminada y guardada en: {output_path}")

print("\n✅ Todos los PDFs traducidos y guardados correctamente.")
