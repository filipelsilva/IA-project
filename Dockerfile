FROM python:3.8-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /test

RUN pip install numpy
