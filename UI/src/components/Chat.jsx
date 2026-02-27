import { useState, useRef } from "react";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [pdfInfo, setPdfInfo] = useState(null);

  // Stable session id (VERY IMPORTANT)
  const userIdRef = useRef("user-" + crypto.randomUUID());
  const fileInputRef = useRef(null);

  // ---------------------------
  // Send chat message
  // ---------------------------
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userIdRef.current,
          message: input,
        }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.reply },
      ]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "❌ Server error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------
  // Upload PDF
  // ---------------------------
  const uploadPdf = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append("user_id", userIdRef.current);
    formData.append("file", file);

    setUploading(true);

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/chat`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setPdfInfo(data);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `📄 PDF uploaded successfully. You can now ask questions about it.`,
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "❌ Failed to upload PDF.",
        },
      ]);
    } finally {
      setUploading(false);
      fileInputRef.current.value = "";
    }
  };

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", fontFamily: "Arial" }}>
      <h2>🧠 RAG Agentic Chatbot</h2>

      {/* -------- PDF Upload Section -------- */}
      <div
        style={{
          border: "1px dashed #aaa",
          padding: 12,
          marginBottom: 15,
          borderRadius: 6,
        }}
      >
        <b>Upload PDF (for RAG)</b>
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={(e) => uploadPdf(e.target.files[0])}
          />
          {uploading && <span>⏳ Indexing PDF...</span>}
        </div>

        {pdfInfo && (
          <div style={{ fontSize: 12, marginTop: 6, color: "#555" }}>
            Indexed <b>{pdfInfo.chunks}</b> chunks from{" "}
            <b>{pdfInfo.filename}</b>
          </div>
        )}
      </div>

      {/* -------- Chat Window -------- */}
      <div
        style={{
          border: "1px solid #ccc",
          padding: 10,
          height: 420,
          overflowY: "auto",
          marginBottom: 10,
          borderRadius: 6,
        }}
      >
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <b>{m.role === "user" ? "You" : "Bot"}:</b> {m.content}
          </div>
        ))}
        {loading && <div><i>Bot is typing...</i></div>}
      </div>

      {/* -------- Input -------- */}
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something..."
          style={{ flex: 1, padding: 8 }}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}
