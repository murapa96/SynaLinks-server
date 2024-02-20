ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.3.1
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}
# Set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt update --fix-missing && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
   build-essential \
   python3 \
   python3-dev \
   python3-pip
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "9999"]