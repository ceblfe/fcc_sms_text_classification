import json

# Ruta al archivo notebook
notebook_path = "fcc_sms_text_classification_CBF.ipynb"

# Cargar el notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Recorremos las celdas y corregimos el campo 'name' si falta
for cell in notebook.get("cells", []):
    if "metadata" not in cell:
        cell["metadata"] = {}
    if "name" not in cell["metadata"]:
        cell["metadata"]["name"] = ""

# Guardar el notebook corregido
fixed_path = "fcc_sms_text_classification_CBF_fixed.ipynb"
with open(fixed_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook corregido guardado como: {fixed_path}")
