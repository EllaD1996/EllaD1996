FROM tensorflow/tensorflow:2.10.1
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY selector selector
COPY setup.py setup.py
RUN pip install -e .
CMD uvicorn selector.api.fast_api:app --host 0.0.0.0 --port $PORT
