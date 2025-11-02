# ================================================================
# agregar_fin_seccion.py — Añade [FIN_SECCION] al final de cada línea que tenga "SECCION"
# ================================================================
import os

# Ruta del archivo de entrada y salida
input_file = "dataset_completo.txt"      # ← cambia este nombre si tu archivo se llama distinto
output_file = "secciones_fin.txt"

# Leer y procesar el archivo línea por línea
with open(input_file, "r", encoding="utf-8") as f:
    lineas = f.readlines()

nuevas_lineas = []
for linea in lineas:
    linea = linea.rstrip("\n")  # elimina salto de línea
    if linea.strip().startswith("SECCION "):
        # Solo agrega [FIN_SECCION] si no lo tiene ya
        if not linea.endswith("[FIN_SECCION]"):
            linea = f"{linea} [FIN_SECCION]"
    nuevas_lineas.append(linea)

# Guardar resultado
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(nuevas_lineas))

print(f"✅ Proceso completado. Archivo generado: {output_file}")
