FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-distutils \
    ca-certificates curl tzdata sudo \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' trader

WORKDIR /opt/app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r requirements.txt

# — ib-gateway-docker (headless IB Gateway + IBC) —
ENV IB_GATEWAY_VERSION=1019
ENV IB_GATEWAY_REGULAR_TZ=America/New_York
ENV TRADING_MODE=paper
ENV IBGateway_USER=""
ENV IBGateway_PASSWORD=""
ENV VNC_SERVER_PASSWORD=vncpass
ENV VNC_SERVER_ENABLED=yes
ENV JAVA_HEAP_XMX=512m
ENV GATEWAY_PORT=4002

EXPOSE 4002 5900

ENTRYPOINT ["/entrypoint.sh"]
