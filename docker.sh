#!env sh
# docker run -it -e PYTHONDONTWRITEBYTECODE=1 -w /test -v $PWD:/test python:3.8 bash
docker run -it -w /test -v "$PWD":/test python:3.8 bash
