# nanoserve Production Deployment Guide

This guide details containerization, Kubernetes orchestration, Prometheus metrics scraping, and high-availability configuration for `nanoserve`.

---

## 1. Docker Container Deployment

### Production Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition and install dependencies
COPY pyproject.toml README.md ./
COPY src/ src/

RUN uv pip install --system -e .

EXPOSE 8000

ENV NANOSERVE_HOST="0.0.0.0"
ENV NANOSERVE_PORT="8000"

CMD ["python", "-m", "nanoserve.server.app"]
```

### Running the Container:
```bash
docker build -t nanoserve:1.0.0 .
docker run --gpus all -p 8000:8000 nanoserve:1.0.0
```

---

## 2. Kubernetes Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nanoserve-deployment
  labels:
    app: nanoserve
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nanoserve
  template:
    metadata:
      labels:
        app: nanoserve
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: nanoserve
        image: nanoserve:1.0.0
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: nanoserve-service
spec:
  type: ClusterIP
  selector:
    app: nanoserve
  ports:
  - port: 8000
    targetPort: 8000
```

---

## 3. Prometheus Metrics & Alerting Rules

### Prometheus Scrape Config
```yaml
scrape_configs:
  - job_name: 'nanoserve'
    scrape_interval: 5s
    static_configs:
      - targets: ['nanoserve-service:8000']
```

### Prometheus Alerting Rules (`nanoserve_alerts.yml`)
```yaml
groups:
- name: nanoserve_alerts
  rules:
  - alert: HighKVMemoryUtilization
    expr: nanoserve_gpu_kv_utilization > 0.90
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "GPU KV cache memory utilization above 90%"
      description: "KV cache memory pool is nearing exhaustion, preemption may occur."

  - alert: FrequentPreemptions
    expr: rate(nanoserve_num_preemptions_total[1m]) > 5
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High preemption rate detected"
      description: "Sequences are frequently being swapped or recomputed due to memory pressure."
```
