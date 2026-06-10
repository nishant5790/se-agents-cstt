FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1



WORKDIR /app

# System deps: build tools for PyMuPDF + ffmpeg/ffprobe for the video/audio tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY agent_team/certs/ /usr/local/share/ca-certificates/extra/
RUN if ls /usr/local/share/ca-certificates/extra/*.crt >/dev/null 2>&1; then \
        update-ca-certificates; \
    fi

# Point Python/requests/azure-core at the system CA bundle that now includes the extras.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install Python deps
COPY agent_team/requirements.txt ./agent_team/requirements.txt
RUN pip install --no-cache-dir \
    --timeout 120 --retries 10 \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r agent_team/requirements.txt

# COPY agent_team/requirements.txt ./agent_team/requirements.txt
# RUN pip install -r agent_team/requirements.txt

# App code (sources/models/outputs are mounted at runtime via docker-compose).
COPY agent_team/ ./agent_team/

EXPOSE 8000
CMD ["uvicorn", "agent_team.server:app", "--host", "0.0.0.0", "--port", "8000"]
