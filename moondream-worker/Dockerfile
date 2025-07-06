# ---------- Base image ----------
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app
ENV TRANSFORMERS_CACHE=/app/hf_cache \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------- Replace Torch with the *nightly* that has enable_gqa ----------
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir --pre \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu121

# ---------- App code ----------
COPY handler.py .

CMD ["python", "-u", "handler.py"]
