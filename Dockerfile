FROM python:3.8-bullseye

WORKDIR /app

ENV QA_HOST=0.0.0.0 \
	QA_PORT=8000 \
    ENABLE_PROOFING=1 \
	NUM_RANK=5 \
	BOT_TOKEN=""

ENV VIRTUAL_ENV=/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
	
COPY . .
RUN pip install --no-cache-dir -e .
RUN python console/train.py

CMD ["echo", "Replace me with python web.py or python bot.py"]
