FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
RUN python scripts/run_all.py
EXPOSE 8000 8501
# API:       docker run -p 8000:8000 IMAGE uvicorn app.api:app --host 0.0.0.0 --port 8000
# Dashboard: docker run -p 8501:8501 IMAGE streamlit run app/dashboard.py --server.port 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
