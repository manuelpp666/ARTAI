# ================================================================
# compactar_secciones.py — Unifica cada SECCION en una sola línea
# ================================================================
import re

# 🔹 Archivos de entrada/salida
input_file = "dataset_completo.txt"       # <-- cámbialo por tu archivo actual
output_file = "dataset_compactado.txt"    # <-- se guardará el texto limpio

def compactar_secciones(texto):
    """
    Une todas las líneas pertenecientes a una misma sección
    en una sola línea, separadas por espacios.
    """
    # Dividir el texto en secciones según el patrón SECCION ...
    bloques = re.split(r"(?=SECCION\s+[^\n]+?\s+SECCION\s+)", texto)
    secciones = []

    for bloque in bloques:
        bloque = bloque.strip()
        if not bloque:
            continue

        # Quitar saltos de línea dentro de la sección
        bloque = re.sub(r"\s*\n\s*", " ", bloque)
        # Normalizar espacios múltiples
        bloque = re.sub(r"\s{2,}", " ", bloque)

        secciones.append(bloque.strip())

    return "\n".join(secciones)


if __name__ == "__main__":
    with open(input_file, "r", encoding="utf-8") as f:
        texto = f.read()

    resultado = compactar_secciones(texto)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(resultado)

    print(f"✅ Dataset compactado guardado en '{output_file}'")
