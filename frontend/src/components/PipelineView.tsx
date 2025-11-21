import React from "react";

const steps = [
  "Load config",
  "Prepare data preview",
  "Load base and fine-tuned models",
  "Run evaluation",
  "Build / load RAG index",
  "Chat with the tutor"
];

const PipelineView: React.FC = () => {
  return (
    <section>
      <h2>Pipeline</h2>
      <p className="small-text">
        This is a read-only overview of the LangGraph workflow that runs on the backend.
      </p>
      <ol style={{ paddingLeft: "1.25rem", margin: "0.75rem 0 0.5rem" }}>
        {steps.map((s, i) => (
          <li key={s} style={{ fontSize: "0.9rem", marginBottom: "0.25rem" }}>
            {i + 1}. {s}
          </li>
        ))}
      </ol>
      <p className="small-text">
        In a live interview, you can run the full pipeline from the terminal and use this UI mainly for
        evaluation and chat.
      </p>
    </section>
  );
};

export default PipelineView;
