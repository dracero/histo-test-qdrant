/**
 * RAG Histología Neo4j + A2UI — Frontend Logic
 * ==============================================
 * Chat client with image upload, markdown rendering, 
 * A2UI JSON visualization, and temario panel.
 */

const API_BASE = '';

// ── State ───────────────────────────────────────────────────────────
let pendingImageBase64 = null;
let pendingImageName = null;
let isWaiting = false;

// ── DOM refs ────────────────────────────────────────────────────────
const messagesContainer = document.getElementById('messages-container');
const welcomeScreen = document.getElementById('welcome-screen');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// ── Init ────────────────────────────────────────────────────────────
checkStatus();
setInterval(checkStatus, 15000);

async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        if (data.ready) {
            statusDot.classList.add('online');
            statusText.textContent = `${data.n_temas} temas · ${data.device}`;
            if (data.imagen_activa) {
                statusText.textContent += ` · 📌 ${data.imagen_activa}`;
            }
        } else {
            statusDot.classList.remove('online');
            statusText.textContent = data.error || 'Inicializando...';
        }
    } catch {
        statusDot.classList.remove('online');
        statusText.textContent = 'Sin conexión';
    }
}

// ── Send message ────────────────────────────────────────────────────
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query || isWaiting) return;

    // Hide welcome
    if (welcomeScreen) welcomeScreen.style.display = 'none';

    // Add user message
    addMessage('user', query, pendingImageBase64);

    // Clear input
    chatInput.value = '';
    autoResize(chatInput);
    isWaiting = true;
    sendBtn.disabled = true;

    // Show typing
    const typingEl = showTyping();

    try {
        const body = { query };
        if (pendingImageBase64) {
            body.image_base64 = pendingImageBase64;
            body.image_filename = pendingImageName;
        }

        // Single request — avoid duplicate RAG execution
        const chatRes = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        removeTyping(typingEl);

        console.log("Response status:", chatRes.status);
        if (!chatRes.ok) {
            const err = await chatRes.json().catch(() => ({ detail: 'Error desconocido' }));
            console.error("Chat error:", err);
            addMessage('assistant', `❌ Error: ${err.detail || chatRes.statusText}`);
        } else {
            const data = await chatRes.json();
            console.log("Chat data received:", data);
            addMessage('assistant', data.respuesta, null, data);
        }

    } catch (err) {
        removeTyping(typingEl);
        addMessage('assistant', `❌ Error de conexión: ${err.message}`);
    }

    // Cleanup
    removeImage();
    isWaiting = false;
    sendBtn.disabled = false;
    checkStatus();
}

// ── Send chip ───────────────────────────────────────────────────────
function sendChip(el) {
    chatInput.value = el.textContent;
    sendMessage();
}

// ── Keyboard ────────────────────────────────────────────────────────
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ── Image handling ──────────────────────────────────────────────────
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const base64Full = e.target.result;
        // Strip "data:image/xxx;base64," prefix
        pendingImageBase64 = base64Full.split(',')[1];
        pendingImageName = file.name;

        // Show preview
        document.getElementById('image-thumb').src = base64Full;
        document.getElementById('image-name').textContent = file.name;
        document.getElementById('image-size').textContent = formatBytes(file.size);
        document.getElementById('image-preview-bar').classList.add('visible');
    };
    reader.readAsDataURL(file);

    // Reset file input so same file can be selected again
    event.target.value = '';
}

function removeImage() {
    pendingImageBase64 = null;
    pendingImageName = null;
    document.getElementById('image-preview-bar').classList.remove('visible');
}

async function limpiarImagen() {
    try {
        await fetch(`${API_BASE}/api/imagen/limpiar`, { method: 'POST' });
        removeImage();
        checkStatus();
    } catch { }
}

// ── Message rendering ───────────────────────────────────────────────
function addMessage(role, text, imageBase64 = null, metadata = null) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Image thumbnail for user messages
    if (imageBase64 && role === 'user') {
        const img = document.createElement('img');
        img.className = 'message-image-preview';
        img.src = `data:image/png;base64,${imageBase64}`;
        bubble.appendChild(img);
    }

    // Render text
    if (role === 'assistant') {
        bubble.innerHTML += renderMarkdown(text);
    } else {
        const p = document.createElement('p');
        p.textContent = text;
        bubble.appendChild(p);
    }

    // Image gallery from database (retrieved images)
    if (metadata && metadata.imagenes_recuperadas && metadata.imagenes_recuperadas.length > 0) {
        const gallery = document.createElement('div');
        gallery.className = 'image-gallery';

        const galleryTitle = document.createElement('div');
        galleryTitle.className = 'gallery-title';
        galleryTitle.textContent = `📸 Imágenes del manual (${metadata.imagenes_recuperadas.length})`;
        gallery.appendChild(galleryTitle);

        const galleryGrid = document.createElement('div');
        galleryGrid.className = 'gallery-grid';

        metadata.imagenes_recuperadas.forEach((img, idx) => {
            const item = document.createElement('div');
            item.className = 'gallery-item';

            const imgEl = document.createElement('img');
            imgEl.className = 'gallery-img';
            imgEl.src = img.url;
            imgEl.alt = img.etiqueta || img.nombre_archivo || `Imagen ${idx + 1}`;
            imgEl.loading = 'lazy';
            imgEl.onclick = () => openLightbox(img.url, img.etiqueta, img.caption);
            item.appendChild(imgEl);

            if (img.etiqueta || img.nombre_archivo) {
                const label = document.createElement('div');
                label.className = 'gallery-label';
                label.textContent = img.etiqueta || img.nombre_archivo;
                item.appendChild(label);
            }

            if (img.caption) {
                const caption = document.createElement('div');
                caption.className = 'gallery-caption';
                caption.textContent = img.caption.substring(0, 120) + (img.caption.length > 120 ? '...' : '');
                item.appendChild(caption);
            }

            galleryGrid.appendChild(item);
        });

        gallery.appendChild(galleryGrid);
        bubble.appendChild(gallery);
    }

    wrapper.appendChild(bubble);

    // Metadata
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    const now = new Date();
    meta.textContent = now.toLocaleTimeString('es-AR', { hour: '2-digit', minute: '2-digit' });

    if (metadata) {
        if (metadata.estructura_identificada) {
            meta.textContent += ` · 🏷️ ${metadata.estructura_identificada}`;
        }
        if (metadata.imagen_activa) {
            meta.textContent += ` · 📌 ${metadata.imagen_activa}`;
        }
        const tray = metadata.trayectoria || [];
        const finalNode = tray.find(t => t.tiempo_total !== undefined);
        if (finalNode) {
            meta.textContent += ` · ⏱️ ${finalNode.tiempo_total}s`;
        }
    }

    wrapper.appendChild(meta);
    messagesContainer.appendChild(wrapper);
    scrollToBottom();
}

function showTyping() {
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.id = 'typing-indicator';
    div.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    messagesContainer.appendChild(div);
    scrollToBottom();
    return div;
}

function removeTyping(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// ── Simple Markdown renderer ────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Tables (process before other rules)
    html = html.replace(/^(\|.+\|)\n(\|[-:| ]+\|)\n((?:\|.+\|\n?)*)/gm, (match, header, sep, body) => {
        const headers = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`);
        const rows = body.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`);
            return `<tr>${cells.join('')}</tr>`;
        });
        return `<table><thead><tr>${headers.join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
    });

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold + Italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Lists
    html = html.replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    // Remove nested <ul> wraps
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Numbered lists
    html = html.replace(/^\s*\d+\. (.+)$/gm, '<li>$1</li>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not already structured
    if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
    }

    return html;
}

// ── Temario ─────────────────────────────────────────────────────────
let temarioLoaded = false;

async function toggleTemario() {
    const overlay = document.getElementById('temario-overlay');
    const panel = document.getElementById('temario-panel');
    const isVisible = overlay.classList.contains('visible');

    if (isVisible) {
        overlay.classList.remove('visible');
        panel.style.display = 'none';
    } else {
        overlay.classList.add('visible');
        panel.style.display = 'flex';
        if (!temarioLoaded) await loadTemario();
    }
}

async function loadTemario() {
    const list = document.getElementById('temario-list');
    try {
        const res = await fetch(`${API_BASE}/api/temario`);
        const data = await res.json();
        if (data.temas && data.temas.length > 0) {
            list.innerHTML = data.temas.map((t, i) =>
                `<div class="temario-item" onclick="sendChip(this)">
                    <span class="num">${i + 1}.</span> ${escapeHtml(t)}
                </div>`
            ).join('');
            temarioLoaded = true;
        } else {
            list.innerHTML = '<p style="color:var(--text-muted)">Sin temas disponibles</p>';
        }
    } catch {
        list.innerHTML = '<p style="color:var(--error)">Error cargando temario</p>';
    }
}


// ── Utils ───────────────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Lightbox for database images ────────────────────────────────────
function openLightbox(url, title, caption) {
    // Remove existing lightbox if any
    const existing = document.getElementById('image-lightbox');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'image-lightbox';
    overlay.className = 'lightbox-overlay';
    overlay.onclick = (e) => { if (e.target === overlay) closeLightbox(); };

    const container = document.createElement('div');
    container.className = 'lightbox-container';

    // Close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'lightbox-close';
    closeBtn.innerHTML = '✕';
    closeBtn.onclick = closeLightbox;
    container.appendChild(closeBtn);

    // Image
    const img = document.createElement('img');
    img.className = 'lightbox-img';
    img.src = url;
    img.alt = title || 'Imagen histológica';
    container.appendChild(img);

    // Title
    if (title) {
        const titleEl = document.createElement('div');
        titleEl.className = 'lightbox-title';
        titleEl.textContent = title;
        container.appendChild(titleEl);
    }

    // Caption
    if (caption) {
        const captionEl = document.createElement('div');
        captionEl.className = 'lightbox-caption';
        captionEl.textContent = caption;
        container.appendChild(captionEl);
    }

    overlay.appendChild(container);
    document.body.appendChild(overlay);

    // Close on Escape
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closeLightbox();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
}

function closeLightbox() {
    const lb = document.getElementById('image-lightbox');
    if (lb) lb.remove();
}


// ── Log ─────────────────────────────────────────────────────────────
console.log('🔬 RAG Histología Neo4j + A2UI client initialized');
