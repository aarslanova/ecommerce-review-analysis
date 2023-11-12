FROM python:3.10.12-slim

WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry

RUN poetry install --without dev

COPY app.py model.pth ./

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "app.py"]