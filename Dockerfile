FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_DEBUG=false
ENV PORT=5000
EXPOSE 5000

CMD ["python", "app_v2.py"]
