ARG CUDA_VERSION=10.2
ARG OS_VERSION=18.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

RUN apt update && apt upgrade -y

RUN apt install python3-pip python3 python3-dev libxml2-dev libxslt-dev libprotobuf-dev protobuf-compiler -y
RUN pip3 install --upgrade pip
RUN pip3 install setuptools==53.0.0

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /workspace