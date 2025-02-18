FROM python:3.12-alpine 

WORKDIR /app

COPY . .

RUN python -m pip install --no-cache-dir . && rm -rf /app && mkdir /app