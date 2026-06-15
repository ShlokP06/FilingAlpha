import { useQuery } from "@tanstack/react-query";
import {
  Card,
  Title,
  Text,
  BarChart,
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";
import { fetchPredictions, parsePredictions } from "@/api";
import type { ParsedPrediction, ModelMetrics } from "@/api";
import { EmptyState } from "@/components/EmptyState";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorBanner } from "@/components/ErrorBanner";

// Metric display names for common keys
const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  f1: "F1 Score",
  f1_score: "F1 Score",
  precision: "Precision",
  recall: "Recall",
  roc_auc: "ROC-AUC",
  auc: "AUC",
  mse: "MSE",
  mae: "MAE",
  rmse: "RMSE",
  r2: "R²",
  ic: "IC",
  sharpe: "Sharpe",
};

function friendlyMetricName(key: string): string {
  return METRIC_LABELS[key.toLowerCase()] ?? key;
}

function formatMetricValue(value: string | number | undefined): string {
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  // Show as percentage if plausibly a 0-1 proportion metric
  if (value >= 0 && value <= 1) return value.toFixed(4);
  return value.toFixed(3);
}

interface MetricsTableProps {
  metrics: ModelMetrics;
}

function MetricsTable({ metrics }: MetricsTableProps) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return <Text className="text-slate-500 text-sm">No metrics available.</Text>;
  }
  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell className="text-slate-400">Metric</TableHeaderCell>
          <TableHeaderCell className="text-slate-400 text-right">Value</TableHeaderCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {entries.map(([k, v]) => (
          <TableRow key={k} className="hover:bg-slate-800/50">
            <TableCell className="text-slate-300">{friendlyMetricName(k)}</TableCell>
            <TableCell className="text-right font-mono text-slate-200">
              {formatMetricValue(v)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

interface PredictionPanelProps {
  prediction: ParsedPrediction;
  index: number;
}

function PredictionPanel({ prediction, index }: PredictionPanelProps) {
  const featureData = prediction.feature_importances.slice(0, 15).map((fi) => ({
    feature: fi.feature,
    Importance: Number(fi.importance.toFixed(4)),
  }));

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h3 className="text-base font-semibold text-slate-200">
          {index === 0 ? "Latest Model" : `Model ${index + 1}`}: {prediction.model_type}
        </h3>
        <span className="text-xs text-slate-500">
          {new Date(prediction.created_at).toLocaleString()}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <Title className="text-slate-200 mb-3">Walk-Forward Metrics</Title>
          <MetricsTable metrics={prediction.metrics} />
        </Card>

        {featureData.length > 0 && (
          <Card className="bg-slate-900 border-slate-800">
            <Title className="text-slate-200 mb-3">Feature Importances</Title>
            <Text className="text-slate-500 text-xs mb-4">Top {featureData.length} features</Text>
            <BarChart
              data={featureData}
              index="feature"
              categories={["Importance"]}
              colors={["blue"]}
              layout="vertical"
              yAxisWidth={120}
              className="h-72"
            />
          </Card>
        )}
      </div>
    </div>
  );
}

export function ModelView() {
  const predictionsQuery = useQuery({
    queryKey: ["predictions"],
    queryFn: fetchPredictions,
  });

  if (predictionsQuery.isPending) return <LoadingSpinner />;
  if (predictionsQuery.isError) {
    return (
      <ErrorBanner
        message={
          predictionsQuery.error instanceof Error
            ? predictionsQuery.error.message
            : "Failed to load model predictions"
        }
      />
    );
  }

  const rawPredictions = predictionsQuery.data ?? [];
  const predictions = parsePredictions(rawPredictions);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-100 mb-1">Model Performance</h2>
        <p className="text-sm text-slate-500">
          Walk-forward prediction results. Metrics and feature importances are parsed from the
          stored JSON. The most recently created model run is shown first.
        </p>
      </div>

      {predictions.length === 0 ? (
        <EmptyState
          title="No model results yet"
          description="Run the prediction pipeline to populate model metrics and feature importances."
        />
      ) : (
        <div className="space-y-8">
          {predictions.map((p, i) => (
            <PredictionPanel key={`${p.model_type}-${p.created_at}`} prediction={p} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
