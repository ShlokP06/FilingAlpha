import { useState } from "react";
import { NavBar } from "@/components/NavBar";
import type { View } from "@/components/NavBar";
import { SignalExplorer } from "@/views/SignalExplorer";
import { BacktestView } from "@/views/BacktestView";
import { ModelView } from "@/views/ModelView";

function App() {
  const [activeView, setActiveView] = useState<View>("signals");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <NavBar activeView={activeView} onViewChange={setActiveView} />
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeView === "signals" && <SignalExplorer />}
        {activeView === "backtest" && <BacktestView />}
        {activeView === "model" && <ModelView />}
      </main>
    </div>
  );
}

export default App;
