import { useQuery } from "@tanstack/react-query";
import {
  Card,
  Title,
  Text,
  Metric,
  BadgeDelta,
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";
import { fetchBacktests, bestBacktest } from "@/api";
import type { BacktestResult } from "@/api";
import { EmptyState } from "@/components/EmptyState";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorBanner } from "@/components/ErrorBanner";

function fmt(value: number, decimals = 3): string {
  return value.toFixed(decimals);
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function TStatBadge({ tstat }: { tstat: number }) {
  const abs = Math.abs(tstat);
  if (abs >= 1.96) {
    return (
      <BadgeDelta deltaType="increase" size="xs">
        {fmt(tstat, 2)}
      </BadgeDelta>
    );
  }
  if (abs >= 1.0) {
    return (
      <BadgeDelta deltaType="moderateIncrease" size="xs">
        {fmt(tstat, 2)}
      </BadgeDelta>
    );
  }
  return (
    <BadgeDelta deltaType="unchanged" size="xs">
      {fmt(tstat, 2)}
    </BadgeDelta>
  );
}

interface SummaryCardProps {
  label: string;
  value: string;
  sub: string;
}

function SummaryCard({ label, value, sub }: SummaryCardProps) {
  return (
    <Card className="bg-slate-900 border-slate-800">
      <Text className="text-slate-400">{label}</Text>
      <Metric className="text-slate-100 mt-1">{value}</Metric>
      <Text className="text-slate-500 text-xs mt-1">{sub}</Text>
    </Card>
  );
}

function BestSignalPanel({ best }: { best: BacktestResult }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
      <SummaryCard
        label="Best Signal"
        value={best.signal}
        sub={`${best.horizon_days}d horizon`}
      />
      <SummaryCard
        label="IC"
        value={fmt(best.ic, 3)}
        sub="Information Coefficient"
      />
      <SummaryCard
        label="IC t-stat"
        value={fmt(best.ic_tstat, 2)}
        sub={Math.abs(best.ic_tstat) >= 1.96 ? "Statistically significant" : "Below 1.96 threshold"}
      />
      <SummaryCard
        label="L/S Sharpe"
        value={fmt(best.ls_sharpe, 2)}
        sub={`Hit rate ${fmtPct(best.hit_rate)}`}
      />
    </div>
  );
}

export function BacktestView() {
  const backtestsQuery = useQuery({
    queryKey: ["backtests"],
    queryFn: fetchBacktests,
  });

  if (backtestsQuery.isPending) return <LoadingSpinner />;
  if (backtestsQuery.isError) {
    return (
      <ErrorBanner
        message={
          backtestsQuery.error instanceof Error
            ? backtestsQuery.error.message
            : "Failed to load backtest results"
        }
      />
    );
  }

  const results = backtestsQuery.data ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-100 mb-1">Backtest Results</h2>
        <p className="text-sm text-slate-500">
          Signal IC and long-short performance across horizons. Results are reported as-is —
          classical NLP signals on small samples often yield modest ICs, which is expected.
        </p>
      </div>

      {results.length === 0 ? (
        <EmptyState
          title="No backtest results yet"
          description="Run the backtesting pipeline to populate results."
        />
      ) : (
        <>
          {(() => {
            const best = bestBacktest(results);
            return best !== null ? <BestSignalPanel best={best} /> : null;
          })()}

          <Card className="bg-slate-900 border-slate-800">
            <Title className="text-slate-200 mb-4">Signal x Horizon Performance</Title>
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeaderCell className="text-slate-400">Signal</TableHeaderCell>
                  <TableHeaderCell className="text-slate-400 text-right">Horizon</TableHeaderCell>
                  <TableHeaderCell className="text-slate-400 text-right">IC</TableHeaderCell>
                  <TableHeaderCell className="text-slate-400 text-right">IC t-stat</TableHeaderCell>
                  <TableHeaderCell className="text-slate-400 text-right">L/S Sharpe</TableHeaderCell>
                  <TableHeaderCell className="text-slate-400 text-right">Hit Rate</TableHeaderCell>
                  <TableHeaderCell className="text-slate-400 text-right">Cum. Return</TableHeaderCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((r, i) => (
                  <TableRow key={i} className="hover:bg-slate-800/50">
                    <TableCell className="font-mono text-slate-200 text-sm">{r.signal}</TableCell>
                    <TableCell className="text-right text-slate-300">{r.horizon_days}d</TableCell>
                    <TableCell className="text-right font-mono text-slate-300">
                      {fmt(r.ic)}
                    </TableCell>
                    <TableCell className="text-right">
                      <TStatBadge tstat={r.ic_tstat} />
                    </TableCell>
                    <TableCell className="text-right font-mono text-slate-300">
                      {fmt(r.ls_sharpe, 2)}
                    </TableCell>
                    <TableCell className="text-right text-slate-300">
                      {fmtPct(r.hit_rate)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-slate-300">
                      {fmtPct(r.cum_return)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>

          <p className="text-xs text-slate-600">
            IC t-stat badges: green = |t| &ge; 1.96, yellow = |t| &ge; 1.0, gray = below
            threshold. Signals with weak ICs are included for transparency.
          </p>
        </>
      )}
    </div>
  );
}
