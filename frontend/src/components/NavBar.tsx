type View = "signals" | "backtest" | "model";

interface NavBarProps {
  activeView: View;
  onViewChange: (view: View) => void;
}

const NAV_ITEMS: { id: View; label: string }[] = [
  { id: "signals", label: "Signal Explorer" },
  { id: "backtest", label: "Backtest Results" },
  { id: "model", label: "Model Performance" },
];

export function NavBar({ activeView, onViewChange }: NavBarProps) {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 flex items-center gap-8 h-14">
        <span className="text-sm font-semibold tracking-widest text-slate-400 uppercase mr-4">
          FilingAlpha
        </span>
        <nav className="flex gap-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={[
                "px-4 py-1.5 rounded text-sm font-medium transition-colors",
                activeView === item.id
                  ? "bg-blue-600 text-white"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800",
              ].join(" ")}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}

export type { View };
