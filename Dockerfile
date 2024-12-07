# Gunakan image Python official
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependensi
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Salin kode aplikasi ke container
COPY . .

# Set environment variable untuk Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Expose port untuk aplikasi
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "app.py"]