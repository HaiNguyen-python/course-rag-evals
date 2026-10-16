FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]"
COPY corpus ./corpus
COPY evals ./evals
EXPOSE 8000
CMD ["sh", "-c", "course-rag index && course-rag serve --port 8000"]
