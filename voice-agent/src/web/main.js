const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
const host = `${protocol}://${window.location.host}`;

const pttBtn = document.getElementById('ptt-btn');
const statusText = document.getElementById('status-text');
const transcriptLog = document.getElementById('transcript-log');
const llmLog = document.getElementById('llm-log');
const chatLog = document.getElementById('chat-log');
const outputDeviceSelect = document.getElementById('output-device');
const modelInput = document.getElementById('ollama-model');
const ttsSelect = document.getElementById('tts-backend');
const speakPartialsCheckbox = document.getElementById('speak-partials');
const saveSettingsBtn = document.getElementById('save-settings');
const repeatBtn = document.getElementById('repeat-btn');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');

let micSocket = null;
let eventsSocket = null;
let audioContext = null;
let processor = null;
let mediaStream = null;
let capturing = false;
let partialElement = null;
let assistantElement = null;

function updateStatus(text) {
  statusText.textContent = text;
}

async function connectEvents() {
  eventsSocket = new WebSocket(`${host}/ws/events`);
  eventsSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch (data.type) {
      case 'partial':
        renderPartial(data.text);
        break;
      case 'final':
        renderFinal(data.text);
        break;
      case 'llm':
        renderLLMToken(data.token);
        break;
      case 'status':
        updateStatus(data.message);
        break;
      default:
        console.debug('unknown event', data);
    }
  };
  eventsSocket.onopen = () => updateStatus('Connected');
  eventsSocket.onclose = () => updateStatus('Events disconnected');
}

function renderPartial(text) {
  if (!partialElement) {
    partialElement = document.createElement('div');
    partialElement.className = 'partial';
    transcriptLog.appendChild(partialElement);
    transcriptLog.scrollTop = transcriptLog.scrollHeight;
  }
  partialElement.textContent = text;
}

function renderFinal(text) {
  if (partialElement) {
    transcriptLog.removeChild(partialElement);
    partialElement = null;
  }
  const entry = document.createElement('div');
  entry.className = 'final';
  entry.textContent = text;
  transcriptLog.appendChild(entry);
  chatLog.appendChild(entry.cloneNode(true));
  transcriptLog.scrollTop = transcriptLog.scrollHeight;
  chatLog.scrollTop = chatLog.scrollHeight;
  assistantElement = null;
}

function renderLLMToken(token) {
  if (!assistantElement) {
    assistantElement = document.createElement('div');
    assistantElement.className = 'final';
    llmLog.appendChild(assistantElement);
    const chatEntry = assistantElement.cloneNode(false);
    chatLog.appendChild(chatEntry);
    assistantElement._chatRef = chatEntry;
  }
  assistantElement.textContent += token;
  if (assistantElement._chatRef) {
    assistantElement._chatRef.textContent = assistantElement.textContent;
  }
  llmLog.scrollTop = llmLog.scrollHeight;
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function startCapture() {
  if (capturing) return;
  capturing = true;
  pttBtn.classList.add('active');
  updateStatus('Listening...');
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
      },
    });
  }
  audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    if (!micSocket || micSocket.readyState !== WebSocket.OPEN) return;
    const channelData = event.inputBuffer.getChannelData(0);
    const pcm = floatTo16BitPCM(channelData);
    micSocket.send(pcm);
  };
  source.connect(processor);
  const gain = audioContext.createGain();
  gain.gain.value = 0;
  processor.connect(gain);
  gain.connect(audioContext.destination);
  micSocket = new WebSocket(`${host}/ws/mic`);
  micSocket.binaryType = 'arraybuffer';
  micSocket.onopen = () => updateStatus('Streaming audio...');
  micSocket.onclose = () => updateStatus('Mic closed');
}

async function stopCapture() {
  if (!capturing) return;
  capturing = false;
  pttBtn.classList.remove('active');
  updateStatus('Processing...');
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (micSocket && micSocket.readyState === WebSocket.OPEN) {
    micSocket.close();
  }
  micSocket = null;
}

function floatTo16BitPCM(input) {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

async function loadDevices() {
  const resp = await fetch('/api/audio/devices');
  const data = await resp.json();
  outputDeviceSelect.innerHTML = '';
  data.devices.forEach((device) => {
    const option = document.createElement('option');
    option.value = device.name;
    option.textContent = device.name + (device.is_default ? ' (default)' : '');
    outputDeviceSelect.appendChild(option);
  });
}

async function loadHealth() {
  const resp = await fetch('/api/health');
  const data = await resp.json();
  if (data.device) {
    outputDeviceSelect.value = data.device;
  }
  modelInput.value = data.ollama_model;
  ttsSelect.value = data.tts_backend;
  speakPartialsCheckbox.checked = data.speak_partials;
}

async function saveSettings() {
  const payload = {
    output_device: outputDeviceSelect.value,
    ollama_model: modelInput.value,
    tts_backend: ttsSelect.value,
    speak_partials: speakPartialsCheckbox.checked,
  };
  await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  updateStatus('Settings saved');
}

async function repeatLast() {
  await fetch('/api/repeat', { method: 'POST' });
}

async function sendChat() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = '';
  await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}

function setupPTT() {
  pttBtn.addEventListener('mousedown', startCapture);
  pttBtn.addEventListener('mouseup', stopCapture);
  pttBtn.addEventListener('mouseleave', stopCapture);
  pttBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startCapture();
  });
  pttBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopCapture();
  });

  window.addEventListener('keydown', (event) => {
    if (event.code === 'Space' && !event.repeat && document.activeElement === document.body) {
      startCapture();
    }
  });
  window.addEventListener('keyup', (event) => {
    if (event.code === 'Space') {
      stopCapture();
    }
  });
}

function setupButtons() {
  saveSettingsBtn.addEventListener('click', saveSettings);
  repeatBtn.addEventListener('click', repeatLast);
  chatSend.addEventListener('click', sendChat);
  chatInput.addEventListener('keyup', (event) => {
    if (event.key === 'Enter') {
      sendChat();
    }
  });
}

async function init() {
  setupPTT();
  setupButtons();
  await connectEvents();
  await loadDevices();
  await loadHealth();
}

init().catch((err) => console.error(err));
