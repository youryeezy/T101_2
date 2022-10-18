FROM python:3.8-slim-buster

RUN mkdir app

COPY ./ ./app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python3", "generator.py"]