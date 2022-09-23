FROM jayaneetha/images:tf2.1.0-gpu-py3.6.8-base

WORKDIR /app

COPY ./requirements/req_fastapi.txt requirements.txt
COPY ./requirements/req_tf.txt .
COPY ./requirements/req.txt .

USER root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y update
RUN apt-get install -y libsndfile1
RUN apt-get install -y ffmpeg

USER user

RUN pip install --no-cache-dir --upgrade -r req_tf.txt
RUN pip install --no-cache-dir --upgrade -r req.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

ADD ./FastAPIBackend ./FastAPIBackend
ADD ./rl ./rl
ADD ./ZetaPolicy ./ZetaPolicy
ADD ./logging.conf .

RUN mkdir media
RUN mkdir persistent_store

CMD ["uvicorn", "FastAPIBackend.main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api"]