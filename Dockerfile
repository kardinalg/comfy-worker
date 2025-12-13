
# 1️⃣ База — офіційний comfyui-api образ від Salad
FROM ghcr.io/saladtechnologies/comfyui-api:comfy0.3.27-torch2.6.0-cuda12.4-runtime

# 2️⃣ Environment variables (дефолти)
ENV TMP_DIR=/tmp/comfy_worker
ENV WORKFLOWS_DIR=/opt/comfy_workflows
ENV PYTHONUNBUFFERED=1

# 3️⃣ Створюємо потрібні директорії
RUN mkdir -p \
    ${TMP_DIR} \
    ${WORKFLOWS_DIR} \
    /opt/worker

# 4️⃣ Копіюємо workflow-и
COPY workflows/ ${WORKFLOWS_DIR}/

# 5️⃣ Копіюємо worker
COPY worker.py /opt/worker/worker.py

# 6️⃣ (опційно) Python залежності для worker
# Якщо requirements.txt порожній — просто не додавай ці рядки
COPY requirements.txt /opt/worker/requirements.txt
RUN if [ -s /opt/worker/requirements.txt ]; then \
        pip install --no-cache-dir -r /opt/worker/requirements.txt; \
    fi

# 7️⃣ entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 8️⃣ comfyui-api слухає 3000 порт
EXPOSE 3000

# 9️⃣ Старт
CMD ["/entrypoint.sh"]
