import React, { useState } from "react";

type Props = {
  evalSummary: string | null;
  onEvalSummary: (summary: string | null) => void;
};

const API_BASE = "http://127.0.0.1:8000";

const EvalPanel: React.FC<Props> = ({ evalSummary, onEvalSummary }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runEval = async () => {
    setIsRunning(true);
    setError(null);
    onEvalSummary(null);

    try {
      const res = await fetch(`${API_BASE}/run-eval`, {
        method: "POST"
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      const summary = [
        `Base score:      ${data.base_score?.toFixed?.(3) ?? data.base_score}`,
        `Finetuned score: ${data.finetuned_score?.toFixed?.(3) ?? data.finetuned_score}`,
        `Samples:         ${data.num_samples}`,
        data.results_path ? `Results file:   ${data.results_path}` : ""
      ]
        .filter(Boolean)
        .join("\n");

      onEvalSummary(summary);
    } catch (err: any) {
      setError(err?.message ?? "Failed to run evaluation.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <section>
      <h2>Evaluation</h2>
      <p className="small-text">
        Runs a tiny eval set against base and fine-tuned models using the backend pipeline.
      </p>
      <div className="button-row">
        <button className="button" onClick={runEval} disabled={isRunning}>
          {isRunning ? "Running..." : "Run Evaluation"}
        </button>
      </div>

      {error && (
        <p className="small-text" style={{ color: "#b91c1c" }}>
          Error: {error}
        </p>
      )}

      {evalSummary && (
        <pre className="mono" style={{ marginTop: "0.5rem" }}>
          {evalSummary}
        </pre>
      )}
    </section>
  );
};

export default EvalPanel;
