import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  Title,
  LineChart,
  Select,
  SelectItem,
  Text,
} from "@tremor/react";
import { fetchCompanies, fetchSignals } from "@/api";
import { EmptyState } from "@/components/EmptyState";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorBanner } from "@/components/ErrorBanner";

export function SignalExplorer() {
  const [selectedTicker, setSelectedTicker] = useState<string>("");

  const companiesQuery = useQuery({
    queryKey: ["companies"],
    queryFn: fetchCompanies,
  });

  const signalsQuery = useQuery({
    queryKey: ["signals", selectedTicker],
    queryFn: () => fetchSignals(selectedTicker),
    enabled: selectedTicker !== "",
  });

  if (companiesQuery.isPending) return <LoadingSpinner />;
  if (companiesQuery.isError) {
    return (
      <ErrorBanner
        message={
          companiesQuery.error instanceof Error
            ? companiesQuery.error.message
            : "Failed to load companies"
        }
      />
    );
  }

  const companies = companiesQuery.data ?? [];

  if (companies.length === 0) {
    return (
      <EmptyState
        title="No companies found"
        description="Run the pipeline to populate companies and signals."
      />
    );
  }

  const chartData =
    signalsQuery.data?.points.map((p) => ({
      date: p.filing_date,
      "LM Negative": Number(p.lm_negative.toFixed(4)),
      "LM Uncertainty": Number(p.lm_uncertainty.toFixed(4)),
      "LM Litigious": Number(p.lm_litigious.toFixed(4)),
    })) ?? [];

  const yoySimilarityData =
    signalsQuery.data?.points.map((p) => ({
      date: p.filing_date,
      "YoY Similarity": Number(p.yoy_similarity.toFixed(4)),
    })) ?? [];

  const riskData =
    signalsQuery.data?.points.map((p) => ({
      date: p.filing_date,
      "Risk Factor Delta": Number(p.risk_factor_delta.toFixed(4)),
    })) ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-100 mb-1">Signal Explorer</h2>
        <p className="text-sm text-slate-500">
          NLP-derived signals from SEC filings per ticker. LM lexicon scores, year-over-year
          document similarity, and risk factor change.
        </p>
      </div>

      <div className="max-w-xs">
        <label className="block text-xs font-medium text-slate-400 mb-1 uppercase tracking-wide">
          Ticker
        </label>
        <Select
          value={selectedTicker}
          onValueChange={(v) => setSelectedTicker(v)}
          placeholder="Select a ticker..."
        >
          {companies.map((c) => (
            <SelectItem key={c.ticker} value={c.ticker}>
              {c.ticker} — {c.name}
            </SelectItem>
          ))}
        </Select>
      </div>

      {selectedTicker === "" && (
        <EmptyState
          title="Select a ticker to view signals"
          description="Choose a company above to load its filing signal time series."
        />
      )}

      {selectedTicker !== "" && signalsQuery.isPending && <LoadingSpinner />}

      {selectedTicker !== "" && signalsQuery.isError && (
        <ErrorBanner
          message={
            signalsQuery.error instanceof Error
              ? signalsQuery.error.message
              : "Failed to load signals"
          }
        />
      )}

      {selectedTicker !== "" && signalsQuery.isSuccess && chartData.length === 0 && (
        <EmptyState
          title="No signal data for this ticker"
          description="The pipeline has not yet produced signals for this company."
        />
      )}

      {chartData.length > 0 && (
        <div className="space-y-5">
          <Card className="bg-slate-900 border-slate-800">
            <Title className="text-slate-200">LM Tone Scores</Title>
            <Text className="text-slate-500 mb-4">
              Loughran-McDonald lexicon: fraction of negative, uncertainty, and litigious words per
              filing.
            </Text>
            <LineChart
              data={chartData}
              index="date"
              categories={["LM Negative", "LM Uncertainty", "LM Litigious"]}
              colors={["red", "amber", "orange"]}
              yAxisWidth={64}
              connectNulls
              className="h-56"
            />
          </Card>

          <Card className="bg-slate-900 border-slate-800">
            <Title className="text-slate-200">Year-over-Year Document Similarity</Title>
            <Text className="text-slate-500 mb-4">
              Cosine similarity between the current filing and prior-year filing (TF-IDF). Higher =
              less change.
            </Text>
            <LineChart
              data={yoySimilarityData}
              index="date"
              categories={["YoY Similarity"]}
              colors={["blue"]}
              yAxisWidth={64}
              connectNulls
              className="h-56"
            />
          </Card>

          <Card className="bg-slate-900 border-slate-800">
            <Title className="text-slate-200">Risk Factor Delta</Title>
            <Text className="text-slate-500 mb-4">
              Change in risk factor section length relative to prior year. Positive = expanded risk
              disclosures.
            </Text>
            <LineChart
              data={riskData}
              index="date"
              categories={["Risk Factor Delta"]}
              colors={["violet"]}
              yAxisWidth={64}
              connectNulls
              className="h-56"
            />
          </Card>
        </div>
      )}
    </div>
  );
}
