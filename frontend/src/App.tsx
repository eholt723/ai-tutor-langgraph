import React, { useState } from "react";
import PipelineView from "./components/PipelineView";
import EvalPanel from "./components/EvalPanel";
import ChatPanel from "./components/ChatPanel";

const App: React.FC = () => {
  const [evalSummary, setEvalSummary] = useState<string | null>(null);

  return (
    <div className="app-root">
      <header className="app-header">
        <h1 className="app-title">AI Tutor LangGraph</h1>
        <p className="app-subtitle">
          Fine-tuning, evaluation, and RAG in a small, local-friendly workflow.
        </p>
      </header>

      <main className="app-main">
        <div className="card">
          <PipelineView />
        </div>

        <div className="card">
          <EvalPanel onEvalSummary={setEvalSummary} evalSummary={evalSummary} />
        </div>

        <div className="card">
          <ChatPanel />
        </div>
      </main>
    </div>
  );
};

export default App;
