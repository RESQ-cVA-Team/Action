FROM rasa/rasa-sdk:3.20.1@sha256:7c3fb13a2a3791bf0575ca9ea1eeb3d68b2e9dfa518025db26418d8126a1530b

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

COPY requirements.txt requirements-dev.txt .
RUN /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements-dev.txt

EXPOSE 5055

ENTRYPOINT []
CMD ["sleep", "infinity"]