FROM node:22-alpine AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY index.html vite.config.js ./
COPY src ./src
COPY config ./config
COPY public ./public
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The engine (engine/) runs in this container too, as a background job started from /admin.
# It needs PyTorch: the CPU-only build keeps the image near 2 GB. INSTALL_ENGINE=0 leaves it out,
# which gives a small image that serves the library and stores uploads but cannot process them.
ARG INSTALL_ENGINE=0
COPY engine/requirements.txt engine/requirements.txt
RUN if [ "$INSTALL_ENGINE" = "1" ]; then \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r engine/requirements.txt && \
      pip uninstall -y opencv-python && \
      pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless ; \
    fi
COPY engine ./engine
# Fetch the pose and upscaling weights now, so a job never waits on a download.
RUN if [ "$INSTALL_ENGINE" = "1" ]; then cd engine && RELENTLESS_DEVICE=cpu python -c \
      "from relentless import people; from relentless.upscale import Upscaler; people.pose_model(); Upscaler('general')" ; \
    fi

COPY backend ./backend
COPY config ./config
COPY data ./data
COPY --from=frontend /app/dist ./dist

# Uploads, runs and the job list live in /data. Mount a volume there to keep them across deploys.
# No GPU here: the engine runs on the CPU with a smaller pose input, and in low-memory mode (person
# tracking in a child process that exits before upscaling starts). RELENTLESS_THREADS caps its threads.
ENV PEELSET_DATA=/data RELENTLESS_DEVICE=cpu RELENTLESS_POSE_IMGSZ=960 RELENTLESS_POSE_BATCH=2 \
    RELENTLESS_LOW_MEMORY=1 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1
RUN mkdir -p /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
