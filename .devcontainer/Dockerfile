FROM rasa/rasa-sdk:3.19.0@sha256:39ddf4e684d300f43cdbfbde3d1f445b0a9d64110944f54e36388738486db743

USER root

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pyright \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt

EXPOSE 5055

ENTRYPOINT []
CMD ["sleep", "infinity"]