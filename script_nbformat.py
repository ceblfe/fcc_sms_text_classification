import nbformat

# Cargar el notebook
notebook_path = "fcc_sms_text_classification_CBF_fixed.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Guardar el notebook corregido
fixed_path = "fcc_sms_text_classification_CBF_fixed_2.ipynb"
with open(fixed_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook corregido guardado como: {fixed_path}")
