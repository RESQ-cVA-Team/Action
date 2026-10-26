FROM rasa/rasa-sdk:3.21.0@sha256:4a80e0e80b4e7c9af9e57ed78387dba24e92dfcfad4e5ccd7353e12c587405b3 AS deps

USER root

WORKDIR /build

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /tmp/wheels -r requirements.txt


FROM rasa/rasa-sdk:3.21.0@sha256:4a80e0e80b4e7c9af9e57ed78387dba24e92dfcfad4e5ccd7353e12c587405b3

USER root

# OS packages get security patches independently of the pinned image tag.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# The base image's system Python has its own pip, outside /opt/venv; the
# certifi it vendors is scanned like any installed package.
RUN /usr/bin/python3 -m pip install --no-cache-dir --break-system-packages --root-user-action=ignore --upgrade pip==26.2.2

COPY requirements.txt VERSION .
COPY --from=deps /tmp/wheels /tmp/wheels

RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt \
	&& rm -rf /tmp/wheels

WORKDIR /app

RUN mkdir -p /app && chown -R 1001:1001 /app

ARG ACTION_VERSION=""
ARG ACTION_COMMIT_SHA=""
ARG ACTION_IMAGE_TAG=""
ARG ACTION_BUILD_DATE=""
ARG ACTION_SSOT_VERSION=""

ENV ACTION_VERSION=${ACTION_VERSION}
ENV ACTION_COMMIT_SHA=${ACTION_COMMIT_SHA}
ENV ACTION_IMAGE_TAG=${ACTION_IMAGE_TAG}
ENV ACTION_BUILD_DATE=${ACTION_BUILD_DATE}
ENV ACTION_SSOT_VERSION=${ACTION_SSOT_VERSION}

LABEL org.opencontainers.image.version=${ACTION_VERSION}
LABEL org.opencontainers.image.revision=${ACTION_COMMIT_SHA}
LABEL org.opencontainers.image.created=${ACTION_BUILD_DATE}

COPY --chown=1001:1001 src/ ./src
COPY --chown=1001:1001 rasa_sdk_plugins/ ./rasa_sdk_plugins

# --- Build-time assertion: ensure SSOT YAML files are present ---
# If this fails, you likely forgot:
#   git submodule update --init --recursive
# before running docker build. Alternatively vendor the files or clone the SSOT repo here.
RUN test -f src/shared/SSOT/ChartType.yml \
	|| (echo "\n[ERROR] SSOT YAML files missing in image.\n" \
			"Did you run 'git submodule update --init --recursive' before 'docker build'?\n" \
			"Contents of src/shared/SSOT (if any):"; ls -al src/shared/SSOT || true; exit 1)

EXPOSE 5055

USER 1001

ENTRYPOINT ["python", "-m", "rasa_sdk", "--actions", "src.actions"]
