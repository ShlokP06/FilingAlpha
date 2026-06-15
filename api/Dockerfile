FROM python:3.11-slim

WORKDIR /app

# Install only the runtime dependencies needed by the API.
# Omitting heavy ML/pipeline packages (edgartools, yfinance, scikit-learn, etc.)
# keeps the image small.  Use `pip install .` instead if you want exact
# pinned versions from pyproject.toml at the cost of a larger layer.
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    "sqlalchemy>=2.0" \
    "psycopg[binary]>=3.2" \
    "pydantic>=2.9" \
    "pydantic-settings>=2.5" \
    "prometheus-client>=0.21"

# Copy only the packages the API needs at runtime.
COPY core/ core/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
