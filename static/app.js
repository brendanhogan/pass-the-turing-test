// Pass The Turing Test — WebSocket client and renderers

const state = {
  agents: [],                  // [{name, bio, color, alive}]
  selectedAgent: null,
  thoughtsByAgent: {},         // {name: [{round, phase, text}]}
  votesByAgent: {},            // {name: [{round, target}]}
  messagesByRoom: {},          // {room_key: [{speaker, text, round}]}
  round: 0,
  phase: null,
  eventLog: [],                // raw events in arrival order, for replay
  gameEnded: false,
  isReplaying: false,
};

const $ = (id) => document.getElementById(id);

const PHASE_LABELS = {
  'discussion': 'GROUP CHAT',
  'confessional': 'CONFESSIONAL BOOTH',
  'dm-pick': 'DM PHASE — PICKING',
  'dm-exchange': 'DM PHASE — TALKING',
  'voting': 'TRIBAL COUNCIL',
  'exit-interview': 'EXIT INTERVIEW',
};

function init() {
  $('replay-btn').addEventListener('click', toggleReplay);
  connect();
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => { if (!state.gameEnded) $('phase').textContent = 'CONNECTED'; };
  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    handleEvent(data);
  };
  ws.onclose = () => {
    if (!state.gameEnded) {
      $('phase').textContent = 'DISCONNECTED';
      setTimeout(connect, 3000);
    }
  };
}

function handleEvent(ev) {
  if (ev.type === 'snapshot') {
    resetDisplayState();
    state.eventLog = [];
    for (const e of ev.events) {
      state.eventLog.push(e);
      processEvent(e);
    }
    // detect if game already ended (we joined late)
    if (state.eventLog.some(e => e.type === 'end')) {
      showReplayMode();
    }
    return;
  }
  // Live event — record then process. (Skip recording during replay so we don't
  // pollute the log with our own replay-driven re-emits.)
  if (!state.isReplaying) state.eventLog.push(ev);
  processEvent(ev);
}

function processEvent(ev) {
  switch (ev.type) {
    case 'state':
      state.agents = ev.agents;
      state.round = ev.round || 0;
      renderAgentGrid();
      $('round-counter').textContent = state.round ? `ROUND ${state.round}` : 'ROUND —';
      if (state.selectedAgent) renderMind();
      return;

    case 'message':
      pushMessage(ev);
      appendChatMessage(ev);
      if (state.selectedAgent && isRoomVisibleToSelected(ev.room)) renderMind();
      if (ev.speaker && ev.speaker !== 'HOST' && ev.speaker !== 'PRODUCER') {
        markThinking(ev.speaker, false);
      }
      return;

    case 'thought':
      pushThought(ev.agent, ev.round, ev.phase, ev.text);
      if (state.selectedAgent === ev.agent) renderMind();
      return;

    case 'phase':
      state.phase = ev.phase;
      setPhase(ev.phase, ev.round);
      appendPhaseDivider(ev.phase);
      return;

    case 'thinking':
      markThinking(ev.agent, true);
      return;

    case 'vote':
      markThinking(ev.agent, false);
      pushVote(ev.agent, ev.round, ev.target);
      appendChatVote(ev);
      if (state.selectedAgent === ev.agent) renderMind();
      return;

    case 'tally':
      appendChatTally(ev);
      return;

    case 'elimination':
      appendChatElimination(ev);
      return;

    case 'end':
      appendChatEnd(ev);
      showReplayMode();
      return;
  }
}

function resetDisplayState() {
  $('chat-feed').innerHTML = '';
  state.thoughtsByAgent = {};
  state.votesByAgent = {};
  state.messagesByRoom = {};
  state.round = 0;
  state.phase = null;
  // reset agents to all alive (they were before the game started)
  state.agents = state.agents.map(a => ({...a, alive: true}));
  renderAgentGrid();
  if (state.selectedAgent) renderMind();
}

function pushMessage(ev) {
  if (!state.messagesByRoom[ev.room]) state.messagesByRoom[ev.room] = [];
  state.messagesByRoom[ev.room].push({
    speaker: ev.speaker, text: ev.text, round: ev.round,
  });
}

function pushThought(name, round, phase, text) {
  if (!state.thoughtsByAgent[name]) state.thoughtsByAgent[name] = [];
  state.thoughtsByAgent[name].push({round, phase, text});
}

function pushVote(name, round, target) {
  if (!state.votesByAgent[name]) state.votesByAgent[name] = [];
  state.votesByAgent[name].push({round, target});
}

function isRoomVisibleToSelected(room) {
  const name = state.selectedAgent;
  if (!name) return false;
  if (room === 'group') return false;
  if (room === `confessional:${name}`) return true;
  if (room.startsWith('dm:')) {
    const members = room.slice(3).split('+');
    return members.includes(name);
  }
  return false;
}

function setPhase(phase, round) {
  const el = $('phase');
  const label = PHASE_LABELS[phase] || phase.toUpperCase();
  el.textContent = `${label} · R${round}`;
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

// ---------- replay ----------

function showReplayMode() {
  state.gameEnded = true;
  $('replay-controls').hidden = false;
  $('live-indicator').classList.add('ended');
  $('live-label').textContent = 'ENDED';
  $('phase').textContent = 'BROADCAST ENDED';
}

async function toggleReplay() {
  if (state.isReplaying) {
    stopReplay();
    return;
  }
  await startReplay();
}

async function startReplay() {
  if (state.eventLog.length === 0) return;
  state.isReplaying = true;
  $('replay-btn').textContent = '■ STOP';
  $('replay-btn').classList.add('is-replaying');
  $('live-indicator').classList.remove('ended');
  $('live-indicator').classList.add('replaying');
  $('live-label').textContent = 'REPLAY';

  resetDisplayState();

  const events = state.eventLog;
  for (let i = 0; i < events.length; i++) {
    if (!state.isReplaying) return;
    const ev = events[i];
    processEvent(ev);
    if (i + 1 < events.length) {
      const speed = parseFloat($('replay-speed').value) || 4;
      const dt = (events[i + 1].t || 0) - (ev.t || 0);
      // Real-time delta, capped to avoid awkward silences, scaled by speed
      const waitMs = Math.min(Math.max(dt, 0) * 1000, 4000) / speed;
      if (waitMs > 5) await sleep(waitMs);
    }
  }
  stopReplay();
}

function stopReplay() {
  state.isReplaying = false;
  $('replay-btn').textContent = '▶ REPLAY';
  $('replay-btn').classList.remove('is-replaying');
  $('live-indicator').classList.remove('replaying');
  $('live-indicator').classList.add('ended');
  $('live-label').textContent = 'ENDED';
  $('phase').textContent = 'BROADCAST ENDED';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---------- agent grid ----------

function renderAgentGrid() {
  const grid = $('agent-grid');
  grid.innerHTML = '';
  for (const agent of state.agents) {
    const card = document.createElement('div');
    card.className = 'agent-card';
    if (!agent.alive) card.classList.add('eliminated');
    if (state.selectedAgent === agent.name) card.classList.add('selected');
    card.dataset.name = agent.name;
    card.innerHTML = `
      <div class="agent-avatar" style="background:${agent.color}">${agent.name[0]}</div>
      <div class="agent-info">
        <div class="agent-name">${escapeHtml(agent.name)}</div>
        <div class="agent-status">${agent.alive ? 'IN PLAY' : 'ELIMINATED'}</div>
      </div>
    `;
    card.addEventListener('click', () => selectAgent(agent.name));
    grid.appendChild(card);
  }
}

function markThinking(name, on) {
  const card = document.querySelector(`.agent-card[data-name="${cssEscape(name)}"]`);
  if (!card) return;
  card.classList.toggle('thinking', on);
}

function selectAgent(name) {
  state.selectedAgent = name;
  renderAgentGrid();
  renderMind();
}

// ---------- subject file ----------

function renderMind() {
  const el = $('mind-content');
  if (!state.selectedAgent) {
    el.innerHTML = `
      <p class="hint">No subject selected.</p>
      <p class="hint">Select a contestant from the left to read their<br/>private thoughts, confessionals, DMs, and votes.</p>
    `;
    return;
  }
  const agent = state.agents.find(a => a.name === state.selectedAgent);
  if (!agent) return;

  const thoughts = (state.thoughtsByAgent[agent.name] || []).slice().reverse();
  const votes = state.votesByAgent[agent.name] || [];

  const confRoom = `confessional:${agent.name}`;
  const confMessages = state.messagesByRoom[confRoom] || [];

  const dmRooms = Object.keys(state.messagesByRoom)
    .filter(r => r.startsWith('dm:') && r.slice(3).split('+').includes(agent.name));

  el.innerHTML = `
    <div class="mind-header">
      <div class="mind-name" style="color:${agent.color}">${escapeHtml(agent.name)}</div>
      <div class="mind-bio">${escapeHtml(agent.bio)}</div>
      <div class="mind-status ${agent.alive ? '' : 'eliminated'}">${agent.alive ? 'IN PLAY' : 'ELIMINATED'}</div>
    </div>

    <div class="mind-section">
      <div class="mind-section-label">Internal Monologue · latest first</div>
      ${thoughts.length === 0
        ? '<p class="hint">No thoughts recorded yet.</p>'
        : thoughts.map(t => `
            <div class="thought-entry">
              <span class="thought-tag">R${t.round} · ${escapeHtml(t.phase)}</span>
              ${escapeHtml(t.text)}
            </div>
          `).join('')}
    </div>

    <div class="mind-section">
      <div class="mind-section-label">Confessional Tapes</div>
      ${renderConfessionals(confMessages)}
    </div>

    <div class="mind-section">
      <div class="mind-section-label">Private DMs</div>
      ${dmRooms.length === 0
        ? '<p class="hint">No DMs yet.</p>'
        : dmRooms.map(room => renderDMThread(room, agent.name)).join('')}
    </div>

    <div class="mind-section">
      <div class="mind-section-label">Vote History</div>
      ${votes.length === 0
        ? '<p class="hint">No votes cast.</p>'
        : votes.map(v => `
            <div class="vote-entry">
              <span class="vote-round">R${v.round}</span>
              <span class="vote-arrow">→</span>
              <span class="vote-target" style="color:${agentColor(v.target)}">${escapeHtml(v.target)}</span>
            </div>
          `).join('')}
    </div>
  `;
}

function renderConfessionals(messages) {
  if (messages.length === 0) return '<p class="hint">None yet.</p>';
  const groups = [];
  let current = null;
  for (const m of messages) {
    if (!current || current.round !== m.round) {
      current = {round: m.round, messages: []};
      groups.push(current);
    }
    current.messages.push(m);
  }
  return groups.map(g => `
    <div class="conf-tape">
      <div class="conf-tape-header">ROUND ${g.round}</div>
      ${g.messages.map(m => `
        <div class="conf-tape-line ${m.speaker === 'PRODUCER' ? 'producer' : 'subject'}">
          <span class="conf-tape-speaker">${escapeHtml(m.speaker)}</span>
          <span class="conf-tape-text">${escapeHtml(m.text)}</span>
        </div>
      `).join('')}
    </div>
  `).join('');
}

function renderDMThread(room, agentName) {
  const members = room.slice(3).split('+');
  const other = members[0] === agentName ? members[1] : members[0];
  const otherColor = agentColor(other);
  const messages = state.messagesByRoom[room] || [];
  const rounds = [...new Set(messages.map(m => m.round))];
  return `
    <div class="dm-thread">
      <div class="dm-thread-header">
        with <span style="color:${otherColor}">${escapeHtml(other)}</span>
        <span class="dm-thread-round">R${rounds.join(', R')}</span>
      </div>
      ${messages.map(m => `
        <div class="dm-thread-msg">
          <span class="dm-thread-speaker" style="color:${agentColor(m.speaker)}">${escapeHtml(m.speaker)}:</span>
          <span class="dm-thread-text">${escapeHtml(m.text)}</span>
        </div>
      `).join('')}
    </div>
  `;
}

// ---------- chat feed renderers ----------

function appendChatMessage(ev) {
  const feed = $('chat-feed');
  if (ev.room === 'group') {
    if (ev.speaker === 'HOST') {
      feed.appendChild(createMsg('msg-host', escapeHtml(ev.text)));
    } else {
      const color = agentColor(ev.speaker);
      feed.appendChild(createMsg('msg-speech', `
        <span class="msg-speaker" style="color:${color}">${escapeHtml(ev.speaker)}</span>
        <span class="msg-text">${escapeHtml(ev.text)}</span>
      `));
    }
  } else if (ev.room.startsWith('confessional:')) {
    const subject = ev.room.slice('confessional:'.length);
    const color = agentColor(subject);
    const isProducer = ev.speaker === 'PRODUCER';
    feed.appendChild(createMsg(`msg-confessional ${isProducer ? 'is-producer' : 'is-subject'}`, `
      <div class="msg-conf-tag" style="border-color:${color}; color:${color}">
        ◉ CONFESSIONAL · <span style="color:${color}">${escapeHtml(subject)}</span>
      </div>
      <div class="msg-conf-body">
        <span class="msg-conf-speaker" style="${isProducer ? '' : `color:${color}`}">${escapeHtml(ev.speaker)}</span>
        <span class="msg-conf-text">${escapeHtml(ev.text)}</span>
      </div>
    `));
  } else if (ev.room.startsWith('dm:')) {
    const members = ev.room.slice(3).split('+');
    const [a, b] = members;
    const aColor = agentColor(a);
    const bColor = agentColor(b);
    const speakerColor = agentColor(ev.speaker);
    feed.appendChild(createMsg('msg-dm', `
      <div class="msg-dm-tag">
        ⇆ DM ·
        <span style="color:${aColor}">${escapeHtml(a)}</span>
        <span class="msg-dm-tag-sep">↔</span>
        <span style="color:${bColor}">${escapeHtml(b)}</span>
      </div>
      <div class="msg-dm-body">
        <span class="msg-dm-speaker" style="color:${speakerColor}">${escapeHtml(ev.speaker)}</span>
        <span class="msg-dm-text">${escapeHtml(ev.text)}</span>
      </div>
    `));
  }
  scrollFeed();
}

function appendPhaseDivider(phase) {
  const label = PHASE_LABELS[phase] || phase.toUpperCase();
  $('chat-feed').appendChild(createMsg('msg-phase', `· ${label} ·`));
  scrollFeed();
}

function appendChatVote(ev) {
  const voterColor = agentColor(ev.agent);
  const targetColor = agentColor(ev.target);
  $('chat-feed').appendChild(createMsg('msg-vote', `
    <span class="msg-vote-voter" style="color:${voterColor}">${escapeHtml(ev.agent)}</span>
    <span class="msg-vote-arrow">VOTES →</span>
    <span class="msg-vote-target" style="color:${targetColor}">${escapeHtml(ev.target)}</span>
    <span class="msg-vote-stmt">"${escapeHtml(ev.statement || '')}"</span>
  `));
  scrollFeed();
}

function appendChatTally(ev) {
  const counts = Object.values(ev.tally);
  const max = counts.length ? Math.max(...counts) : 1;
  $('chat-feed').appendChild(createMsg('msg-tally', `
    <div class="msg-tally-title">VOTE TALLY</div>
    <div class="msg-tally-rows">
      ${Object.entries(ev.tally).map(([name, count]) => `
        <div class="msg-tally-row ${name === ev.eliminated ? 'eliminated' : ''}">
          <span class="msg-tally-name" style="color:${agentColor(name)}">${escapeHtml(name)}</span>
          <span class="msg-tally-bar">
            <span class="msg-tally-bar-fill" style="width:${(count/max)*100}%"></span>
          </span>
          <span class="msg-tally-count">${count}</span>
        </div>
      `).join('')}
    </div>
  `));
  scrollFeed();
}

function appendChatElimination(ev) {
  $('chat-feed').appendChild(createMsg('msg-elimination', escapeHtml(ev.agent)));
  scrollFeed();
}

function appendChatEnd(ev) {
  const names = (ev.winners || []).join(' & ');
  $('chat-feed').appendChild(createMsg('msg-end', `${escapeHtml(names)}<br/>survive.`));
  scrollFeed();
}

function createMsg(className, innerHTML) {
  const el = document.createElement('div');
  el.className = `msg ${className}`;
  el.innerHTML = innerHTML;
  return el;
}

function scrollFeed() {
  const feed = $('chat-feed');
  feed.scrollTop = feed.scrollHeight;
}

function agentColor(name) {
  const agent = state.agents.find(a => a.name === name);
  return agent ? agent.color : 'var(--text)';
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

function cssEscape(str) {
  return String(str).replace(/["\\]/g, '\\$&');
}

init();
