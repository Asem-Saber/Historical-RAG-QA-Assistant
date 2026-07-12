const messagesEl = document.getElementById('chat-messages');
const inputForm = document.getElementById('chat-form');
const inputField = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');

function escapeHtml(str) {
    const el = document.createElement('span');
    el.textContent = str;
    return el.innerHTML;
}

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

    const avatar = role === 'user' ? '\u{13000}' : '\u{13080}';
    div.innerHTML = `<div class="message-avatar">${avatar}</div>`;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    div.appendChild(contentDiv);
    return div;
}

function showLoading() {
    const div = document.createElement('div');
    div.className = 'message message-assistant';
    div.id = 'loading-msg';
    div.innerHTML = `
        <div class="message-avatar">\u{13080}</div>
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

    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'citations-toggle';
    toggleBtn.textContent = `\u{1307F} Sources & Citations (${sources.length})`;
    toggleBtn.addEventListener('click', () => listDiv.classList.toggle('open'));

    const listDiv = document.createElement('div');
    listDiv.className = 'citations-list';

    for (const s of sources) {
        const item = document.createElement('div');
        item.className = 'citation-item';
        const strong = document.createElement('strong');
        strong.textContent = `[${s.citation}]`;
        item.appendChild(strong);
        const preview = s.content.length > 250 ? s.content.slice(0, 250) + '...' : s.content;
        item.appendChild(document.createTextNode(' ' + preview));
        listDiv.appendChild(item);
    }

    citationsDiv.appendChild(toggleBtn);
    citationsDiv.appendChild(listDiv);
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
                    contentEl.textContent = fullText + '◌';
                    scrollToBottom();
                } else if (data.type === 'sources') {
                    contentEl.textContent = fullText;
                    renderSources(data.documents, contentEl);
                    sourcesRendered = true;
                    scrollToBottom();
                } else if (data.type === 'error') {
                    contentEl.textContent = 'Error: ' + data.message;
                } else if (data.type === 'done') {
                    if (!sourcesRendered) {
                        contentEl.textContent = fullText;
                    }
                }
            }
        }

        if (!sourcesRendered && contentEl.childNodes.length === 1) {
            contentEl.textContent = fullText;
        }

    } catch (err) {
        removeLoading();
        const errEl = createMessageEl('assistant', 'Connection error: ' + err.message);
        messagesEl.appendChild(errEl);
    }

    setInputEnabled(true);
    inputField.focus();
    scrollToBottom();
});
