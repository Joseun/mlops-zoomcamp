FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

COPY ./starter.py ./starter.py
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY ./starter.py ./starter.py

CMD [ "python", "./starter.py" ]
