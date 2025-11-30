import React, { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type ChatResponse = {
  question: string;
  answer: string;
  model_type: string;
  used_rag: boolean;
  context_preview?: string | null;
};

const ChatPanel: React.FC = () => {
  const [question, setQuestion] = useState("");
  const [useFinetuned, setUseFinetuned] = useState(true);
  const [useRag, setUseRag] = useState(false);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async () => {
    if (!question.trim()) return;

    setIsSending(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question,
          use_finetuned: useFinetuned,
          use_rag: useRag
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data: ChatResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err?.message ?? "Request failed.");
    } finally {
      setIsSending(false);
    }
  };

  return (
    <section>
      <h2>Chat</h2>
      <p className="small-text">
        Ask the tutor a question and compare behaviors. The backend can use the base model, the
        fine-tuned model, and optional RAG context.
      </p>

      <div style={{ marginTop: "0.5rem" }}>
        <div className="label">Question</div>
        <textarea
          className="textarea"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="Ask about variables, loops, functions..."
        />
      </div>

      <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
        <button
          type="button"
          className={useFinetuned ? "button" : "button-outline"}
          onClick={() => setUseFinetuned(prev => !prev)}
        >
          {useFinetuned ? "Using fine-tuned model" : "Using base model"}
        </button>

        <button
          type="button"
          className={useRag ? "button" : "button-outline"}
          onClick={() => setUseRag(prev => !prev)}
        >
          {useRag ? "RAG: On" : "RAG: Off"}
        </button>

        <button className="button" onClick={send} disabled={isSending}>
          {isSending ? "Sending..." : "Send"}
        </button>
      </div>

      {error && (
        <p className="small-text" style={{ color: "#b91c1c", marginTop: "0.5rem" }}>
          Error: {error}
        </p>
      )}

      {response && (
        <div style={{ marginTop: "0.75rem" }}>
          <div className="small-text">
            Model: <strong>{response.model_type}</strong> | RAG:{" "}
            <strong>{response.used_rag ? "On" : "Off"}</strong>
          </div>
          <div style={{ marginTop: "0.35rem" }}>
            <div className="label">Answer</div>
            <div className="mono">{response.answer}</div>
          </div>
          {response.context_preview && (
            <div style={{ marginTop: "0.5rem" }}>
              <div className="label">Context (preview)</div>
              <div className="mono">{response.context_preview}</div>
            </div>
          )}
        </div>
      )}
    </section>
  );
};

export default ChatPanel;
