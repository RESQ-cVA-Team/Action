FROM rasa/rasa-sdk:3.6.2

USER root

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pyright \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt

EXPOSE 5055

ENTRYPOINT []
CMD ["sleep", "infinity"]