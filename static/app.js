// app.js —— 会话管理 + 调用后端 POST /query（FAQ / RAG 统一入口）

const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const statusTextEl = document.getElementById('statusText');
const convListEl = document.getElementById('convList');
const newChatBtn = document.getElementById('newChat');
const convTitleEl = document.getElementById('convTitle');
const appEl = document.getElementById('app');
const sidebarToggle = document.getElementById('sidebarToggle');
const modeSwitchEl = document.getElementById('modeSwitch');
const sourceSelectEl = document.getElementById('sourceSelect');
const modeHintEl = document.getElementById('modeHint');

const API_URL = '/query';
const SESSION_KEY = 'edurag_session_id';
const MODE_KEY = 'edurag_mode';
const SOURCE_KEY = 'edurag_source';

// 模式说明，与后端 qa_router 的语义一致
const MODE_HINT = {
  auto: 'FAQ 优先，未命中自动转 RAG',
  faq: '只查 FAQ 知识库，速度快',
  rag: '只走 RAG，语义检索 + 大模型生成（较慢）',
};

let busy = false;
let currentSessionId = null;   // 当前会话 id
let conversations = [];        // 会话列表缓存
let mode = localStorage.getItem(MODE_KEY) || 'auto';

/* ---------------- 工具函数 ---------------- */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

/** 复制文本到剪贴板（http 环境下自动降级） */
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (e) {
    /* 走降级方案 */
  }

  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.cssText = 'position:fixed;top:-1000px;left:-1000px;opacity:0;';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch (e) {
    ok = false;
  }
  document.body.removeChild(ta);
  return ok;
}

/** 普通文本 -> HTML：转义 + 行内代码 + 换行 */
function textToHtml(str) {
  return escapeHtml(str)
    .replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/\n/g, '<br>');
}

/** 渲染单个代码块（右上角带复制按钮） */
function buildCodeBlock(lang, code) {
  return `<div class="code-block">` +
    `<div class="code-head">` +
      `<span class="code-lang">${escapeHtml(lang)}</span>` +
      `<button type="button" class="copy-btn">复制</button>` +
    `</div>` +
    `<pre><code>${escapeHtml(code)}</code></pre>` +
    `</div>`;
}

/** 富文本渲染：把 ``` 围栏代码块解析成可复制的代码块 */
function renderRich(text) {
  const src = String(text ?? '');
  const fence = /```([\w#+.\-]*)[ \t]*\n?([\s\S]*?)```/g;

  let html = '';
  let last = 0;
  let m;

  while ((m = fence.exec(src)) !== null) {
    // 代码块之前的普通文本：去掉尾部多余空行，避免视觉上多出空行
    const before = src.slice(last, m.index).replace(/\n+$/, '');
    if (before) html += textToHtml(before);

    const lang = (m[1] || 'code').trim() || 'code';
    const code = m[2].replace(/\n$/, '');
    html += buildCodeBlock(lang, code);

    last = m.index + m[0].length;
  }

  // 最后一段普通文本：去掉开头多余空行
  const tail = src.slice(last).replace(/^\n+/, '');
  if (tail) html += textToHtml(tail);

  return html;
}

/** 生成引擎徽标元素 */
function buildBadge(engine) {
  const span = document.createElement('span');
  const labels = { faq: 'FAQ', rag: 'RAG', smalltalk: '直答' };
  const key = labels[engine] ? engine : 'miss';
  span.className = `badge ${key}`;
  span.textContent = labels[engine] || '未命中';
  return span;
}

/** 从历史消息的 meta 文本里推断引擎（"FAQ · 耗时 0.01 s" -> "faq"） */
function engineFromMeta(meta) {
  const text = String(meta || '');
  if (text.startsWith('FAQ')) return 'faq';
  if (text.startsWith('RAG')) return 'rag';
  if (text.startsWith('直答')) return 'smalltalk';
  return '';
}

function setStatus(type, text) {
  statusEl.classList.remove('busy', 'error');
  if (type === 'busy') statusEl.classList.add('busy');
  if (type === 'error') statusEl.classList.add('error');
  statusTextEl.textContent = text;
}

/**
 * 添加一条消息
 * @param {'user'|'bot'} role
 * @param {string} text
 * @param {{miss?:boolean, err?:boolean, meta?:string, engine?:string, html?:boolean}} opts
 */
function addMessage(role, text, opts = {}) {
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  if (opts.miss) msg.classList.add('miss');
  if (opts.err) msg.classList.add('err');

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '我' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (opts.html) {
    bubble.innerHTML = text;
  } else if (role === 'bot') {
    bubble.classList.add('rich');
    bubble.innerHTML = renderRich(text);
    // 保留原始文本，供“复制全文”按钮使用
    bubble.dataset.raw = String(text ?? '');
  } else {
    bubble.textContent = text;
  }

  const wrap = document.createElement('div');

  if (opts.meta || opts.engine) {
    const meta = document.createElement('div');
    meta.className = 'meta';

    // 引擎徽标：FAQ / RAG / 未命中
    if (role === 'bot' && opts.engine) {
      meta.appendChild(buildBadge(opts.engine));
    }

    const metaText = document.createElement('span');
    metaText.className = 'meta-text';
    metaText.textContent = opts.meta || '';
    // 过程说明（例如「FAQ 未命中，已转 RAG」）作为悬浮提示，不占版面
    if (opts.note) metaText.title = opts.note;
    meta.appendChild(metaText);

    // 机器人消息附带“复制全文”按钮
    if (role === 'bot' && !opts.html && !opts.err && text) {
      const copyAll = document.createElement('button');
      copyAll.type = 'button';
      copyAll.className = 'copy-all';
      copyAll.textContent = '复制';
      meta.appendChild(copyAll);
    }

    wrap.appendChild(bubble);
    wrap.appendChild(meta);
  } else {
    wrap.appendChild(bubble);
  }

  msg.appendChild(avatar);
  msg.appendChild(wrap);
  chatEl.appendChild(msg);
  scrollToBottom();
  return msg;
}

/** 显示“正在检索/生成”的占位消息，返回可用于删除的元素 */
function addTyping(text) {
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  msg.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">
      <span class="typing"><i></i><i></i><i></i></span>
      ${text ? `<span class="typing-text">${escapeHtml(text)}</span>` : ''}
    </div>`;
  chatEl.appendChild(msg);
  scrollToBottom();
  return msg;
}

function setBusy(state) {
  busy = state;
  sendBtn.disabled = state;
  sendBtn.classList.toggle('loading', state);
  inputEl.disabled = state;
  if (state) {
    setStatus('busy', mode === 'rag' ? '生成中…' : '检索中…');
  }
}

/* ---------------- 模式 / 学科 ---------------- */

/** 应用模式：同步按钮选中态、持久化、刷新提示 */
function applyMode(next) {
  mode = ['auto', 'faq', 'rag'].includes(next) ? next : 'auto';
  localStorage.setItem(MODE_KEY, mode);
  Array.from(modeSwitchEl.querySelectorAll('button')).forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  if (modeHintEl) modeHintEl.textContent = MODE_HINT[mode];
  // 学科过滤只对 RAG 生效，仅 FAQ 模式下置灰
  if (sourceSelectEl) sourceSelectEl.disabled = mode === 'faq';
}

modeSwitchEl.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-mode]');
  if (!btn || busy) return;
  applyMode(btn.dataset.mode);
});

sourceSelectEl.addEventListener('change', () => {
  localStorage.setItem(SOURCE_KEY, sourceSelectEl.value || '');
});

/** 拉取后端能力描述，填充学科下拉框 */
async function loadMeta() {
  try {
    const data = await requestJSON('/api/meta');
    const saved = localStorage.getItem(SOURCE_KEY) || '';
    (data.sources || []).forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      sourceSelectEl.appendChild(opt);
    });
    if (saved) sourceSelectEl.value = saved;
  } catch (err) {
    /* 能力描述拉取失败不影响问答，静默处理 */
  }
  applyMode(mode);
}

/* ---------------- 会话管理 ---------------- */

/** 统一请求封装：非 2xx 抛出带 status 的错误 */
async function requestJSON(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.message || `请求失败（HTTP ${resp.status}）`);
    err.status = resp.status;
    err.data = data;
    throw err;
  }
  return data;
}

/** "2026-10-18 11:07:50" -> "10-18 11:07" */
function formatTime(str) {
  return str ? str.slice(5, 16) : '';
}

/** 渲染左侧会话列表 */
function renderConversations(list) {
  conversations = list || [];
  convListEl.innerHTML = '';

  if (!conversations.length) {
    const empty = document.createElement('div');
    empty.className = 'conv-empty';
    empty.textContent = '还没有历史对话';
    convListEl.appendChild(empty);
    return;
  }

  conversations.forEach((c) => {
    const item = document.createElement('div');
    item.className = 'conv-item';
    if (String(c.id) === String(currentSessionId)) item.classList.add('active');
    item.dataset.id = c.id;
    item.title = '双击重命名';

    const info = document.createElement('div');
    info.className = 'conv-info';

    const title = document.createElement('div');
    title.className = 'conv-title';
    title.textContent = c.title || '新对话';

    const sub = document.createElement('div');
    sub.className = 'conv-sub';
    sub.textContent = `${formatTime(c.updated_at)} · ${c.message_count} 条`;

    info.appendChild(title);
    info.appendChild(sub);

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'conv-del';
    del.title = '删除该对话';
    del.textContent = '✕';

    item.appendChild(info);
    item.appendChild(del);
    convListEl.appendChild(item);
  });
}

/** 拉取会话列表 */
async function loadConversations() {
  try {
    const data = await requestJSON('/api/conversations');
    renderConversations(data.conversations || []);
    return conversations;
  } catch (err) {
    setStatus('error', '会话加载失败');
    return [];
  }
}

/** 清空聊天区 */
function clearChat() {
  chatEl.innerHTML = '';
}

/** 顶部欢迎语 */
function showWelcome() {
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  msg.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble rich">
      你好，我是教育知识问答助手，后端接了两套知识库：<br />
      <span class="badge faq">FAQ</span> 命中率优先，检索常见问题的标准答案；<br />
      <span class="badge rag">RAG</span> 语义检索 + 大模型生成，回答需要推理与综述的问题。<br />
      默认「自动」模式：先查 FAQ，未命中自动转 RAG。
    </div>`;
  chatEl.appendChild(msg);
}

/** 新建会话并切换过去 */
async function createConversation() {
  try {
    const data = await requestJSON('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const conv = data.conversation;
    currentSessionId = conv.id;
    localStorage.setItem(SESSION_KEY, currentSessionId);
    convTitleEl.textContent = conv.title || '新对话';

    clearChat();
    showWelcome();
    await loadConversations();
    setStatus('ready', '就绪');
    appEl.classList.remove('sidebar-open');
    inputEl.focus();
  } catch (err) {
    setStatus('error', '新建对话失败');
  }
}

/** 打开（继续）指定会话，加载历史消息 */
async function openConversation(id) {
  if (busy) return;
  try {
    const data = await requestJSON(`/api/conversations/${id}`);
    const conv = data.conversation;

    currentSessionId = conv.id;
    localStorage.setItem(SESSION_KEY, currentSessionId);
    convTitleEl.textContent = conv.title || '新对话';

    clearChat();
    const messages = data.messages || [];
    if (messages.length) {
      messages.forEach((m) => {
        if (m.role === 'user') {
          addMessage('user', m.content);
        } else {
          // 历史消息没有 engine 字段，从 meta 文本前缀还原徽标
          addMessage('bot', m.content, {
            meta: m.meta || '',
            engine: engineFromMeta(m.meta),
            miss: String(m.meta || '').startsWith('未命中'),
          });
        }
      });
      scrollToBottom();
    } else {
      showWelcome();
    }

    renderConversations(conversations); // 刷新选中态
    appEl.classList.remove('sidebar-open');
    inputEl.focus();
  } catch (err) {
    setStatus('error', '会话加载失败');
    await loadConversations();
  }
}

/** 删除会话 */
async function removeConversation(id) {
  if (!confirm('确定删除这条对话吗？删除后无法恢复。')) return;
  try {
    await requestJSON(`/api/conversations/${id}`, { method: 'DELETE' });

    if (String(id) === String(currentSessionId)) {
      currentSessionId = null;
      localStorage.removeItem(SESSION_KEY);
      convTitleEl.textContent = '新对话';
      clearChat();
      showWelcome();
    }

    const list = await loadConversations();
    // 当前没有会话时，自动进入最近的会话或新建一个
    if (currentSessionId === null) {
      if (list.length) {
        await openConversation(list[0].id);
      } else {
        await createConversation();
      }
    }
  } catch (err) {
    setStatus('error', '删除失败');
  }
}

/** 重命名会话 */
async function renameConversation(id, title) {
  if (!title) return;
  try {
    await requestJSON(`/api/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (String(id) === String(currentSessionId)) convTitleEl.textContent = title;
    await loadConversations();
  } catch (err) {
    setStatus('error', '重命名失败');
  }
}

/* ---------------- 核心请求 ---------------- */

async function ask() {
  const query = inputEl.value.trim();
  if (!query || busy) return;

  addMessage('user', query);
  inputEl.value = '';
  autoGrow();

  const typing = addTyping(mode === 'rag' ? '正在生成回答…' : '正在检索…');
  setBusy(true);

  try {
    const data = await requestJSON(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        session_id: currentSessionId,
        mode,
        source: sourceSelectEl.value || '',
      }),
    });
    typing.remove();

    // 首次提问时后端会自动建会话，这里记住会话 id
    if (data.session_id != null) {
      currentSessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, currentSessionId);
    }
    if (data.title) convTitleEl.textContent = data.title;

    const engine = data.engine || (data.found ? 'rag' : '');
    const meta = data.found === false
      ? `未命中 · 耗时 ${data.elapsed ?? '-'} s`
      : `耗时 ${data.elapsed ?? '-'} s`;

    addMessage('bot', data.answer, {
      miss: data.found === false,
      meta,
      engine,
      note: data.note || '',
    });
    setStatus('ready', '就绪');

    // 刷新侧边栏（标题、时间、条数、排序）
    loadConversations();
  } catch (err) {
    typing.remove();
    addMessage('bot', err.message || '请求失败', {
      err: true,
      meta: err.status ? `HTTP ${err.status}` : '',
    });
    setStatus('error', '请求失败');
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

/* ---------------- 输入框行为 ---------------- */

function autoGrow() {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
}

inputEl.addEventListener('input', autoGrow);

inputEl.addEventListener('keydown', (e) => {
  // Enter 发送，Shift + Enter 换行
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    ask();
  }
});

sendBtn.addEventListener('click', ask);

/* ---------------- 复制按钮（事件委托） ---------------- */

chatEl.addEventListener('click', async (e) => {
  const btn = e.target.closest('.copy-btn, .copy-all');
  if (!btn) return;

  let text = '';
  if (btn.classList.contains('copy-btn')) {
    // 复制单个代码块（textContent 已是解码后的原始代码）
    text = btn.closest('.code-block')?.querySelector('code')?.textContent ?? '';
  } else {
    // 复制整条答案
    text = btn.closest('.msg')?.querySelector('.bubble')?.dataset.raw ?? '';
  }
  if (!text) return;

  const ok = await copyText(text);
  const origin = btn.dataset.origin || btn.textContent;
  btn.dataset.origin = origin;
  btn.textContent = ok ? '已复制' : '复制失败';
  btn.classList.toggle('done', ok);
  clearTimeout(btn._timer);
  btn._timer = setTimeout(() => {
    btn.textContent = origin;
    btn.classList.remove('done');
  }, 1400);
});

/* ---------------- 会话列表交互 ---------------- */

newChatBtn.addEventListener('click', createConversation);

convListEl.addEventListener('click', (e) => {
  const item = e.target.closest('.conv-item');
  if (!item) return;

  if (e.target.closest('.conv-del')) {
    e.stopPropagation();
    removeConversation(item.dataset.id);
    return;
  }

  if (String(item.dataset.id) !== String(currentSessionId)) {
    openConversation(item.dataset.id);
  }
});

convListEl.addEventListener('dblclick', (e) => {
  const item = e.target.closest('.conv-item');
  if (!item || e.target.closest('.conv-del')) return;

  const conv = conversations.find((c) => String(c.id) === String(item.dataset.id));
  const name = prompt('重命名对话', conv?.title || '新对话');
  if (name === null) return;
  renameConversation(item.dataset.id, name.trim());
});

// 移动端侧边栏开关
if (sidebarToggle) {
  sidebarToggle.addEventListener('click', () => appEl.classList.toggle('sidebar-open'));
}
appEl.addEventListener('click', (e) => {
  if (e.target === appEl) appEl.classList.remove('sidebar-open');
});

/* ---------------- 启动 ---------------- */

async function init() {
  await loadMeta();
  const list = await loadConversations();
  const saved = localStorage.getItem(SESSION_KEY);

  if (saved && list.some((c) => String(c.id) === String(saved))) {
    // 继续上次打开的对话
    await openConversation(saved);
  } else if (list.length) {
    // 继续最近的对话
    await openConversation(list[0].id);
  } else {
    // 完全没有会话时新建一个
    await createConversation();
  }
}

window.addEventListener('load', () => {
  init();
  inputEl.focus();
});
