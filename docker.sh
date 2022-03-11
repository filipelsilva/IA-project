#!env sh
docker run -it -e PYTHONDONTWRITEBYTECODE=1 --entrypoint=bash -w /test -v "$PWD":/test python:3.8

# Afterwards:
# - pip install numpy
