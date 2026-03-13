// =============================================
//  CONFIG
// =============================================
const API_BASE = "http://127.0.0.1:8000";

// =============================================
//  DOM REFS
// =============================================
const chatMessages = document.getElementById("chatMessages");
const chatContainer = document.getElementById("chatContainer");
const welcomeScreen = document.getElementById("welcomeScreen");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const menuToggle = document.getElementById("menuToggle");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const errorToast = document.getElementById("errorToast");
const suggestions = document.getElementById("suggestions");

// =============================================
//  STATE
// =============================================
let isWaiting = false;

// =============================================
//  TEXTAREA AUTO-RESIZE
// =============================================
userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
    sendBtn.classList.toggle("active", userInput.value.trim().length > 0);
});

// =============================================
//  SEND MESSAGE
// =============================================
function sendMessage(text) {
    const question = text || userInput.value.trim();
    if (!question || isWaiting) return;

    // Hide welcome screen
    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }

    // Add user message
    appendMessage("user", question);

    // Clear input
    userInput.value = "";
    userInput.style.height = "auto";
    sendBtn.classList.remove("active");

    // Show typing indicator
    const typingEl = showTypingIndicator();
    isWaiting = true;

    // Call API
    fetch(`${API_BASE}/chat?question=${encodeURIComponent(question)}`, {
        method: "POST",
    })
        .then((res) => {
            if (!res.ok) throw new Error(`Server error: ${res.status}`);
            return res.json();
        })
        .then((data) => {
            removeTypingIndicator(typingEl);
            if (data.response) {
                appendMessage("assistant", data.response);
            } else if (data.error) {
                showError(data.error);
            }
        })
        .catch((err) => {
            removeTypingIndicator(typingEl);
            showError("Could not connect to the server. Make sure the backend is running.");
            console.error(err);
        })
        .finally(() => {
            isWaiting = false;
            userInput.focus();
        });
}

// =============================================
//  RENDER MESSAGE
// =============================================
function appendMessage(role, content) {
    const msgEl = document.createElement("div");
    msgEl.className = "message";

    const avatarHTML =
        role === "user"
            ? `<div class="message-avatar user">P</div>`
            : `<div class="message-avatar assistant">
           <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
         </div>`;

    const senderName = role === "user" ? "You" : "AI Assistant";

    msgEl.innerHTML = `
    ${avatarHTML}
    <div class="message-body">
      <div class="message-sender">${senderName}</div>
      <div class="message-content">${role === "assistant" ? renderMarkdown(content) : escapeHTML(content)}</div>
    </div>
  `;

    chatMessages.appendChild(msgEl);
    scrollToBottom();
}

// =============================================
//  TYPING INDICATOR
// =============================================
function showTypingIndicator() {
    const el = document.createElement("div");
    el.className = "typing-indicator";
    el.innerHTML = `
    <div class="message-avatar assistant">
      <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
    </div>
    <div class="message-body">
      <div class="message-sender">AI Assistant</div>
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
    chatMessages.appendChild(el);
    scrollToBottom();
    return el;
}

function removeTypingIndicator(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

// =============================================
//  MARKDOWN RENDERER (lightweight)
// =============================================
function renderMarkdown(text) {
    let html = escapeHTML(text);

    // Code blocks: ```...```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code: `...`
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold: **...**
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic: *...*
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Unordered list lines
    html = html.replace(/^[-•] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>");
    // Clean up nested ul tags
    html = html.replace(/<\/ul>\s*<ul>/g, "");

    // Ordered list lines
    html = html.replace(/^\d+\.\s(.+)$/gm, "<li>$1</li>");

    // Paragraphs: split by double newline
    html = html
        .split(/\n{2,}/)
        .map((block) => {
            block = block.trim();
            if (!block) return "";
            if (
                block.startsWith("<pre>") ||
                block.startsWith("<ul>") ||
                block.startsWith("<ol>") ||
                block.startsWith("<li>")
            )
                return block;
            return `<p>${block.replace(/\n/g, "<br>")}</p>`;
        })
        .join("");

    return html;
}

function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// =============================================
//  UTILITY
// =============================================
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

function showError(msg) {
    errorToast.textContent = msg;
    errorToast.classList.add("show");
    setTimeout(() => errorToast.classList.remove("show"), 4000);
}

// =============================================
//  NEW CHAT
// =============================================
function resetChat() {
    // Remove all messages but keep the welcome screen
    chatMessages.innerHTML = "";
    chatMessages.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
    isWaiting = false;
    closeSidebar();
}

// =============================================
//  SIDEBAR TOGGLE (mobile)
// =============================================
function openSidebar() {
    sidebar.classList.add("open");
    sidebarOverlay.classList.add("show");
}

function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("show");
}

// =============================================
//  EVENT LISTENERS
// =============================================

// Send button
sendBtn.addEventListener("click", () => sendMessage());

// Keyboard: Enter to send, Shift+Enter for newline
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Suggestion chips
suggestions.addEventListener("click", (e) => {
    const chip = e.target.closest(".suggestion-chip");
    if (chip) {
        const question = chip.dataset.question;
        sendMessage(question);
    }
});

// New chat button
newChatBtn.addEventListener("click", resetChat);

// Mobile sidebar
menuToggle.addEventListener("click", openSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);

// Focus input on load
userInput.focus();

// =============================================
//  HANDLE URL PARAMS (from Projects page)
// =============================================
(function handleUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const question = params.get("question");
    if (question) {
        // Clean the URL
        window.history.replaceState({}, document.title, window.location.pathname);
        // Auto-send the question after a brief delay
        setTimeout(() => sendMessage(question), 400);
    }
})();
