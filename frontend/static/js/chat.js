const messagesEl = document.getElementById('chat-messages');
const inputForm = document.getElementById('chat-form');
const inputField = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeWelcome() {
    const welcome = document.querySelector('.welcome');
    if (welcome) welcome.remove();
}

function createMessageEl(role, content) {
    const div = document.createElement('div');
    div.className = `message message-${role}`;

    const avatar = role === 'user' ? '𓀀' : '𓂀';
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${content}</div>
    `;
    return div;
}

function showLoading() {
    const div = document.createElement('div');
    div.className = 'message message-assistant';
    div.id = 'loading-msg';
    div.innerHTML = `
        <div class="message-avatar">𓂀</div>
        <div class="message-content">
            <div class="loading">
                <span>Consulting the ancient scrolls</span>
                <div class="loading-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
    `;
    messagesEl.appendChild(div);
    scrollToBottom();
}

function removeLoading() {
    const el = document.getElementById('loading-msg');
    if (el) el.remove();
}

function renderSources(sources, contentEl) {
    if (!sources || sources.length === 0) return;

    const citationsDiv = document.createElement('div');
    citationsDiv.className = 'citations';
    citationsDiv.innerHTML = `
        <button class="citations-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
            𓁿 Sources & Citations (${sources.length})
        </button>
        <div class="citations-list">
            ${sources.map(s => `
                <div class="citation-item">
                    <strong>[${s.citation}]</strong> ${s.content.slice(0, 250)}${s.content.length > 250 ? '...' : ''}
                </div>
            `).join('')}
        </div>
    `;
    contentEl.appendChild(citationsDiv);
}

function setInputEnabled(enabled) {
    inputField.disabled = !enabled;
    sendBtn.disabled = !enabled;
}

function useSuggestion(text) {
    inputField.value = text;
    inputForm.dispatchEvent(new Event('submit'));
}

inputForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = inputField.value.trim();
    if (!query) return;

    removeWelcome();
    setInputEnabled(false);
    inputField.value = '';

    messagesEl.appendChild(createMessageEl('user', query));
    scrollToBottom();
    showLoading();

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        removeLoading();

        const msgEl = createMessageEl('assistant', '');
        messagesEl.appendChild(msgEl);
        const contentEl = msgEl.querySelector('.message-content');
        let fullText = '';
        let sourcesRendered = false;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.type === 'chunk') {
                    fullText += data.content;
                    contentEl.innerHTML = fullText + '<span class="cursor">▌</span>';
                    scrollToBottom();
                } else if (data.type === 'sources') {
                    contentEl.innerHTML = fullText;
                    renderSources(data.documents, contentEl);
                    sourcesRendered = true;
                    scrollToBottom();
                } else if (data.type === 'error') {
                    contentEl.innerHTML = `<em>Error: ${data.message}</em>`;
                } else if (data.type === 'done') {
                    if (!sourcesRendered) {
                        contentEl.innerHTML = fullText;
                    }
                }
            }
        }

        if (!sourcesRendered && contentEl.querySelector('.cursor')) {
            contentEl.innerHTML = fullText;
        }

    } catch (err) {
        removeLoading();
        messagesEl.appendChild(createMessageEl('assistant', `<em>Connection error: ${err.message}</em>`));
    }

    setInputEnabled(true);
    inputField.focus();
    scrollToBottom();
});
