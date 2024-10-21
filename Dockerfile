FROM python:3.8.6

RUN apt-get update
RUN apt-get upgrade -y

WORKDIR /tmp

