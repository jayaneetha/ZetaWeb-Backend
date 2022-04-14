FROM jayaneetha/images:tf2.1.0-gpu-py3.6.8-base

WORKDIR /app

COPY ./requirements/req_fastapi.txt requirements.txt
COPY ./requirements/req_tf.txt .
COPY ./requirements/req.txt .


RUN pip install --no-cache-dir --upgrade -r req_tf.txt
RUN pip install --no-cache-dir --upgrade -r req.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./FastAPIBackend .
COPY ./rl_framework .
COPY ./ZetaPolicy .
COPY ./logging.conf .

RUN mkdir media
RUN mkdir persistent_store

CMD ["uvicorn", "FastAPIBackend.main.app", "--host", "0.0.0.0", "--port", "8000"]