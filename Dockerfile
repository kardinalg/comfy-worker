FROM ghcr.io/saladtechnologies/comfyui-api:comfy0.3.27-torch2.6.0-cuda12.4-runtime

ENV TMP_DIR=/tmp/comfy_worker
ENV WORK_DIR=/opt/worker
ENV MODEL_DIR=/opt/ComfyUI/models

RUN mkdir -p ${TMP_DIR} ${WORK_DIR}

# git + curl, якщо їх немає в базовому образі
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

ADD https://github.com/SaladTechnologies/comfyui-api/releases/download/1.8.2/comfyui-api /comfyui-api
RUN chmod +x /comfyui-api

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000
CMD ["/entrypoint.sh"]
