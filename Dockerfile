FROM rasa/rasa-sdk:3.6.2@sha256:9ebc0abc5b36d9420343a197bdfd9c478155ca86871dc08f8e823c76f484c948

USER root

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY src/ ./src

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
