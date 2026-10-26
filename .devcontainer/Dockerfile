FROM rasa/rasa-sdk:3.21.0@sha256:4a80e0e80b4e7c9af9e57ed78387dba24e92dfcfad4e5ccd7353e12c587405b3

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