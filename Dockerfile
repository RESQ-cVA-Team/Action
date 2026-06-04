FROM rasa/rasa-sdk:3.6.2@sha256:9ebc0abc5b36d9420343a197bdfd9c478155ca86871dc08f8e823c76f484c948 AS deps

USER root

WORKDIR /build

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /tmp/wheels -r requirements.txt


FROM rasa/rasa-sdk:3.6.2@sha256:9ebc0abc5b36d9420343a197bdfd9c478155ca86871dc08f8e823c76f484c948

USER root

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
