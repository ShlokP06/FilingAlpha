/**
 * Typed API client for FilingAlpha backend.
 * Base URL is configured via VITE_API_URL env var (default: http://localhost:8000).
 * All functions throw on non-OK HTTP responses.
 */

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

export interface Company {
  ticker: string;
  cik: string;
  name: string;
  sector: string;
}

export interface SignalPoint {
  filing_date: string;
  fiscal_period: string;
  lm_negative: number;
  lm_uncertainty: number;
  lm_litigious: number;
  yoy_similarity: number;
  risk_factor_delta: number;
  fog_readability: number;
}

export interface SignalSeries {
  ticker: string;
  points: SignalPoint[];
}

export interface BacktestResult {
  signal: string;
  horizon_days: number;
  ic: number;
  ic_tstat: number;
  ls_spread: number;
  spread_tstat: number;
  created_at: string;
}

export interface Prediction {
  model_type: string;
  features_json: string;
  metrics_json: string;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Parsed model types
// ---------------------------------------------------------------------------

export interface ModelMetrics {
  [key: string]: number | string | undefined;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ParsedPrediction {
  model_type: string;
  created_at: string;
  metrics: ModelMetrics;
  feature_importances: FeatureImportance[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`API error ${res.status} fetching ${url}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

function safeParseJson(raw: string): Record<string, unknown> {
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function fetchCompanies(): Promise<Company[]> {
  return apiFetch<Company[]>("/companies");
}

export async function fetchSignals(ticker: string): Promise<SignalSeries> {
  return apiFetch<SignalSeries>(`/signals/${encodeURIComponent(ticker)}`);
}

export async function fetchBacktests(): Promise<BacktestResult[]> {
  return apiFetch<BacktestResult[]>("/backtests");
}

export async function fetchPredictions(): Promise<Prediction[]> {
  return apiFetch<Prediction[]>("/predictions");
}

// ---------------------------------------------------------------------------
// Derived helpers
// ---------------------------------------------------------------------------

/**
 * Parse the raw Prediction records into a structured format.
 * Returns the most recently created entry first.
 */
export function parsePredictions(predictions: Prediction[]): ParsedPrediction[] {
  return predictions
    .slice()
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .map((p) => {
      const metricsRaw = safeParseJson(p.metrics_json);
      const featuresRaw = safeParseJson(p.features_json);

      // Metrics: expect a flat key→number map
      const metrics: ModelMetrics = {};
      for (const [k, v] of Object.entries(metricsRaw)) {
        if (typeof v === "number" || typeof v === "string") {
          metrics[k] = v;
        }
      }

      // Feature importances: either an array or a key→number map
      let feature_importances: FeatureImportance[] = [];
      if (Array.isArray(featuresRaw)) {
        feature_importances = (featuresRaw as unknown[])
          .filter(
            (item): item is { feature: string; importance: number } =>
              typeof item === "object" &&
              item !== null &&
              "feature" in item &&
              "importance" in item,
          )
          .map((item) => ({ feature: String(item.feature), importance: Number(item.importance) }));
      } else {
        feature_importances = Object.entries(featuresRaw)
          .filter(([, v]) => typeof v === "number")
          .map(([k, v]) => ({ feature: k, importance: v as number }))
          .sort((a, b) => b.importance - a.importance);
      }

      return {
        model_type: p.model_type,
        created_at: p.created_at,
        metrics,
        feature_importances,
      };
    });
}

/**
 * Find the best backtest result by IC t-stat magnitude.
 */
export function bestBacktest(results: BacktestResult[]): BacktestResult | null {
  if (results.length === 0) return null;
  return results.reduce((best, cur) =>
    Math.abs(cur.ic_tstat) > Math.abs(best.ic_tstat) ? cur : best,
  );
}
