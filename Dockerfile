# Dockerfile for Fermata Masterclass
#
# Multi-stage build that produces an ~4 GB image carrying:
#   - Python 3.11 + the app's full ML/LLM/API extras
#   - Eclipse Temurin JDK 21 (for Audiveris)
#   - Audiveris 5.6.2 (via the Ubuntu .deb release; works on Debian Bookworm)
#   - ffmpeg + libsndfile (for audio handling)
#
# Build (from repo root):
#   docker build -t fermataacr.azurecr.io/fermata-api:latest .
# or via the deploy script in ~/fermata-hosting/scripts/deploy-app.ps1.

# -----------------------------------------------------------------------------
# Stage 1: builder — installs build tools, compiles wheels.
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Pre-install heavy deps in their own layer so subsequent app-code rebuilds
# don't re-fetch ~3 GB of TF / Torch wheels.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[api,azure,llm,ml]"

# -----------------------------------------------------------------------------
# Stage 2: runtime — slim image with only what's needed at runtime.
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8770 \
    # Storage backend defaults; override in Container App env if needed.
    MASTERCLASS_STORAGE_BACKEND=adls

# System packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        wget \
        # Audiveris .deb declares these as runtime deps
        libfreetype6 \
        libfontconfig1 \
        libxext6 \
        libxrender1 \
        libxtst6 \
        libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Eclipse Temurin JDK 21. We use Adoptium's hotspot tarball, which works on
# Debian Bookworm without needing the ppa.
ARG JDK_URL=https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse
RUN mkdir -p /opt/jdk \
    && wget -qO /tmp/jdk.tar.gz "${JDK_URL}" \
    && tar -xzf /tmp/jdk.tar.gz -C /opt/jdk --strip-components=1 \
    && rm /tmp/jdk.tar.gz
ENV JAVA_HOME=/opt/jdk \
    PATH=/opt/jdk/bin:$PATH

# Audiveris 5.6.2 — the same version we ship to dev boxes. Newer versions
# (5.7+) require Java 24/25 and don't add anything we use.
#
# The .deb declares openjdk-21-jre as a runtime dep. We don't want apt to
# pull in its own JDK on top of the Adoptium one we just installed, so we
# extract the .deb's payload with `dpkg-deb -x` (bypassing the dependency
# manifest) and rely on the AUDIVERIS_HOME env var to point at the lib dir.
ARG AUDIVERIS_URL=https://github.com/Audiveris/audiveris/releases/download/5.6.2/Audiveris-5.6.2-ubuntu22.04-x86_64.deb
RUN wget -qO /tmp/audiveris.deb "${AUDIVERIS_URL}" \
    && mkdir -p /opt \
    && dpkg-deb -x /tmp/audiveris.deb / \
    && rm /tmp/audiveris.deb \
    && test -f /opt/Audiveris/lib/audiveris.jar || ( \
        echo "ERROR: Audiveris jar not at /opt/Audiveris/lib/audiveris.jar after extract"; \
        ls -la /opt/ 2>&1; \
        find /opt -name 'audiveris.jar' 2>/dev/null; \
        exit 1)
ENV AUDIVERIS_HOME=/opt/Audiveris

# Copy installed Python packages from the builder stage.
COPY --from=builder /install /usr/local

# App code.
WORKDIR /app
COPY src ./src
COPY pyproject.toml ./
# Editable install just to register the `masterclass` package; deps already in /usr/local.
RUN pip install --no-deps -e .

# Container Apps expects the app to listen on $PORT.
EXPOSE 8770

# Smoke check on build: ensure key imports succeed.
RUN python -c "import masterclass; import basic_pitch; import torch; print('imports ok')"

# Health endpoint for ACA's HTTP probe (FastAPI returns 200 on /).
CMD ["uvicorn", "masterclass.apps.api.main:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8770", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
