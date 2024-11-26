# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos de tu carpeta al contenedor
COPY . .

# Instala las dependencias listadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que Streamlit usa por defecto
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "Evidencia.py", "--server.port=8501", "--server.address=0.0.0.0"]
