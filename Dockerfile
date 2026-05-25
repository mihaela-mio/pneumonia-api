FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements.txt .

# ✅ IMPORTANT: don't try to uninstall distutils packages
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

ENV PORT=8080

CMD ["gunicorn", "--bind", ":8080", "pneum:app"]
