FROM python:3.13-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x /app/start.sh

EXPOSE 8501 8000

CMD ["/app/start.sh"]