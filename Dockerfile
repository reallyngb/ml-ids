# Phase 13 — Docker deployment
#
# NOTE ON LIVE CAPTURE: Scapy needs raw socket access (CAP_NET_RAW) and
# usually --net=host to see the physical interfaces. This Dockerfile builds
# an image that runs training + the dashboard cleanly; for live_capture,
# run the container as:
#   docker run --net=host --cap-add=NET_RAW --cap-add=NET_ADMIN ml-ids \
#       python src/live/predict_live.py

FROM python:3.11-slim

WORKDIR /app

# System deps: libpcap for scapy packet capture, build tools for xgboost/sklearn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Skip torch in the image by default (large); install separately if you need
# Phase 6 (autoencoder) inside the container.
RUN pip install --no-cache-dir -r requirements.txt --no-deps || true
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost imbalanced-learn \
    matplotlib seaborn scapy pyyaml joblib requests streamlit

COPY . .

RUN mkdir -p data/raw data/processed models/sklearn models/pytorch logs

# Default: run the dashboard. Override CMD to run training or live capture instead.
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
