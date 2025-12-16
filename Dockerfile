FROM ghcr.io/saladtechnologies/comfyui-api:comfy0.3.67-api1.13.4-torch2.8.0-cuda12.8-runtime

ENV TMP_DIR=/tmp/comfy_worker
ENV WORK_DIR=/opt/worker
ENV MODEL_DIR=/opt/ComfyUI/models

RUN mkdir -p ${TMP_DIR} ${WORK_DIR}

# git + curl, якщо їх немає в базовому образі
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN apt-get update && apt-get install -y build-essential

#ADD https://github.com/SaladTechnologies/comfyui-api/releases/download/1.8.2/comfyui-api /comfyui-api
#RUN chmod +x /comfyui-api
ADD https://github.com/SaladTechnologies/comfyui-api/releases/download/1.13.4/comfyui-api /comfyui-api
RUN chmod +x /comfyui-api

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000
CMD ["/entrypoint.sh"]
