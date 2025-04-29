FROM python:3.10

WORKDIR /code

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1 && pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "debug", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "--timeout", "120", "main:app"]
#CMD ["gunicorn",  "--log-level", "debug", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "--timeout", "1600", "main:app"]