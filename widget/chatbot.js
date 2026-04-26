(function () {
  const scriptTag = document.currentScript;
  const API_URL = (scriptTag && scriptTag.dataset.apiUrl) || "http://localhost:8000";

  // --- Styles ---
  const style = document.createElement("style");
  style.textContent = `
    #ms-chatbot-toggle {
      position: fixed; bottom: 24px; right: 24px;
      background: #1a56db; color: #fff; border: none; border-radius: 50%;
      width: 52px; height: 52px; font-size: 24px; cursor: pointer; z-index: 9999;
      box-shadow: 0 2px 8px rgba(0,0,0,.3);
    }
    #ms-chatbot-panel {
      position: fixed; bottom: 88px; right: 24px;
      width: 360px; max-height: 520px;
      background: #fff; border: 1px solid #ddd; border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,.15);
      display: flex; flex-direction: column; z-index: 9999;
      font-family: system-ui, sans-serif; font-size: 14px;
    }
    #ms-chatbot-panel.hidden { display: none; }
    #ms-chat-header {
      background: #1a56db; color: #fff; padding: 12px 16px;
      border-radius: 12px 12px 0 0; font-weight: 600;
    }
    #ms-chat-messages {
      flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px;
    }
    .ms-msg { padding: 8px 12px; border-radius: 8px; max-width: 85%; line-height: 1.4; }
    .ms-msg.user { background: #e8f0fe; align-self: flex-end; }
    .ms-msg.bot  { background: #f3f4f6; align-self: flex-start; }
    .ms-sources { margin-top: 6px; font-size: 12px; }
    .ms-sources a { color: #1a56db; display: block; }
    #ms-chat-input-row {
      display: flex; gap: 8px; padding: 10px;
      border-top: 1px solid #eee;
    }
    #ms-chat-input {
      flex: 1; padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px;
    }
    #ms-chat-send {
      background: #1a56db; color: #fff; border: none; border-radius: 6px;
      padding: 8px 14px; cursor: pointer; font-size: 14px;
    }
    #ms-chat-send:disabled { opacity: .5; cursor: default; }
  `;
  document.head.appendChild(style);

  // --- DOM ---
  const toggle = document.createElement("button");
  toggle.id = "ms-chatbot-toggle";
  toggle.textContent = "💬";
  toggle.title = "Ask Media Suite";

  const panel = document.createElement("div");
  panel.id = "ms-chatbot-panel";
  panel.classList.add("hidden");
  panel.innerHTML = `
    <div id="ms-chat-header">Ask Media Suite</div>
    <div id="ms-chat-messages"></div>
    <div id="ms-chat-input-row">
      <input id="ms-chat-input" type="text" placeholder="Ask a question…" />
      <button id="ms-chat-send">Send</button>
    </div>
  `;

  document.body.appendChild(toggle);
  document.body.appendChild(panel);

  // --- Logic ---
  const messages = document.getElementById("ms-chat-messages");
  const input    = document.getElementById("ms-chat-input");
  const sendBtn  = document.getElementById("ms-chat-send");

  // Conversation history sent to the API on each turn
  const history = [];

  toggle.addEventListener("click", () => panel.classList.toggle("hidden"));

  function addMessage(role, html) {
    const el = document.createElement("div");
    el.className = `ms-msg ${role}`;
    el.innerHTML = html;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  async function send() {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    sendBtn.disabled = true;

    addMessage("user", escapeHtml(q));
    addMessage("bot", "<em>Thinking…</em>");

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, history }),
      });
      const data = await res.json();
      const last = messages.lastElementChild;

      let html = escapeHtml(data.answer).replace(/\n/g, "<br>");
      if (data.sources && data.sources.length) {
        html += `<div class="ms-sources"><strong>Sources:</strong>`;
        data.sources.forEach(s => {
          html += `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title || s.url)}</a>`;
        });
        html += `</div>`;
      }
      last.innerHTML = html;

      // Append this turn to history for the next request
      history.push({ role: "user", content: q });
      history.push({ role: "assistant", content: data.answer });
    } catch (e) {
      messages.lastElementChild.innerHTML = "Sorry, something went wrong. Please try again.";
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
})();
