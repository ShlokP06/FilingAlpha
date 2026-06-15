# FilingAlpha Frontend

React dashboard for visualizing SEC filing NLP signals, backtest results, and model performance.

## Stack

- Vite 6 + React 18 + TypeScript 5 (strict)
- Tailwind CSS 3
- Tremor v3 (data dashboard components backed by Recharts)
- TanStack Query v5 (data fetching + caching)

## Development

```bash
npm install
npm run dev
```

The dev server starts at http://localhost:5173.

## Environment

| Variable       | Default                  | Description                         |
|----------------|--------------------------|-------------------------------------|
| `VITE_API_URL` | `http://localhost:8000`  | Base URL of the FilingAlpha FastAPI |

Create a `.env.local` file to override:

```
VITE_API_URL=http://localhost:8000
```

`VITE_API_URL` is baked in at build time (Vite limitation). For Docker, pass it as a build argument.

## Build

```bash
npm run build      # TypeScript check + Vite build -> dist/
npm run preview    # Preview the production build locally
```

## Docker

```bash
docker build -t filing-alpha-frontend .
docker run -p 5173:80 filing-alpha-frontend
```

Or via docker-compose from the repo root:

```bash
docker-compose up frontend
```

The container serves the static build via nginx on port 80, which docker-compose maps to 5173.

## Views

| View              | Tab       | API endpoint(s)                       |
|-------------------|-----------|---------------------------------------|
| Signal Explorer   | signals   | `/companies`, `/signals/{ticker}`     |
| Backtest Results  | backtest  | `/backtests`                          |
| Model Performance | model     | `/predictions`                        |

All views handle empty/loading/error states gracefully — the UI will not crash when the database is empty.
