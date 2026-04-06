FROM python:3.12

# instalar curl (necessário pro poetry)
RUN apt-get update && apt-get install -y curl

# instalar poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# adicionar poetry ao PATH
ENV PATH="/root/.local/bin:$PATH"

# pasta de trabalho
WORKDIR /app

# copiar arquivos de dependência primeiro (cache docker)
COPY pyproject.toml poetry.lock* ./

# config poetry: instalar deps direto no sistema (sem venv)
RUN poetry config virtualenvs.create false

ENV PIP_DEFAULT_TIMEOUT=120
ENV POETRY_HTTP_TIMEOUT=120

#RUN poetry source add --priority=explicit pytorch-cpu https://download.pytorch.org/whl/cpu  

 # instalar dependências
RUN poetry install --no-interaction --no-ansi --no-root
RUN poetry add torch --source pytorch-cpu
# copiar código
COPY . .

# expor porta
EXPOSE 8000

# rodar fastapi
#CMD ["poetry", "run","python","-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]