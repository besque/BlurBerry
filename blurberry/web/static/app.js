// ── State ─────────────────────────────────────────────────────────────────────
const API          = '';
let stream         = null;
let processing     = false;
let frameId        = 0;
let lastFrameTime  = performance.now();
let fpsSmooth      = 0;
let pendingRequest = false;
let enrolling      = false;   // true while 3-2-1 countdown is running

const stats = { faces: 0, plates: 0, pii: 0, nsfw: 0 };

// ── DOM refs ──────────────────────────────────────────────────────────────────
const videoEl = document.getElementById('videoEl');
const canOut  = document.getElementById('canvasOutput');
const canCap  = document.getElementById('canvasCapture');
const ctx     = canOut.getContext('2d');
const ctxCap  = canCap.getContext('2d');

// ── Status check ──────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch(`${API}/status`);
    const d = await r.json();
    const pill = document.getElementById('statusPill');
    const txt  = document.getElementById('statusText');
    if (d.pipeline_ready) {
      pill.className  = 'status-pill ready';
      txt.textContent = 'pipeline ready';
    } else {
      pill.className  = 'status-pill';
      txt.textContent = 'server up / pipeline loading';
    }
  } catch {
    document.getElementById('statusPill').className      = 'status-pill error';
    document.getElementById('statusText').textContent    = 'server offline';
  }
}
checkStatus();
setInterval(checkStatus, 5000);

// ── Audio event polling (Person 3) ────────────────────────────────────────────
async function pollAudio() {
  try {
    const r = await fetch(`${API}/audio/events`);
    const d = await r.json();
    for (const ev of d.events || []) {
      const types = ev.findings.map(f => f[1]).join(', ');
      logDetection('audio_pii', null, `🔴 Audio PII [${types}]: "${ev.text}"`);
    }
  } catch {}
}
setInterval(pollAudio, 500);

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    videoEl.srcObject     = stream;
    videoEl.style.display = 'none';

    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('startBtn').style.display    = 'none';
    document.getElementById('stopBtn').style.display     = '';
    document.getElementById('snapBtn').style.display     = '';
    document.getElementById('fpsBadge').style.display    = '';
    document.getElementById('enrollHint').style.display  = '';

    // Update enroll sub-hint to show camera is ready
    const sub = document.getElementById('enrollSub');
    sub.textContent = 'Camera live — press E to scan';
    sub.className   = 'enroll-sub ready';

    await new Promise(r => (videoEl.onloadedmetadata = r));
    canOut.width  = canCap.width  = videoEl.videoWidth;
    canOut.height = canCap.height = videoEl.videoHeight;

    processing = true;
    requestAnimationFrame(processLoop);
    toast('Camera started — press E to enroll your face', 'success');
  } catch (e) {
    toast('Camera access denied: ' + e.message, 'error');
  }
}

function stopCamera() {
  processing = false;
  enrolling  = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }

  document.getElementById('placeholder').style.display = '';
  document.getElementById('startBtn').style.display    = '';
  document.getElementById('stopBtn').style.display     = 'none';
  document.getElementById('snapBtn').style.display     = 'none';
  document.getElementById('fpsBadge').style.display    = 'none';
  document.getElementById('enrollHint').style.display  = 'none';

  const sub = document.getElementById('enrollSub');
  sub.textContent = 'Start the camera first';
  sub.className   = 'enroll-sub';

  hideEnrollOverlay();
  ctx.clearRect(0, 0, canOut.width, canOut.height);
  toast('Camera stopped');
}

// ── E key listener ────────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'e' || e.key === 'E') {
    if (processing && !enrolling) startEnroll();
  }
});

// ── Enroll — 3-2-1 countdown then capture ────────────────────────────────────
function startEnroll() {
  enrolling = true;

  const overlay   = document.getElementById('enrollOverlay');
  const countdown = document.getElementById('enrollCountdown');

  overlay.classList.add('active');

  let count = 3;
  countdown.textContent = count;

  const interval = setInterval(() => {
    count--;
    if (count > 0) {
      // Animate number change
      countdown.style.transform = 'scale(1.3)';
      setTimeout(() => { countdown.style.transform = 'scale(1)'; }, 150);
      countdown.textContent = count;
    } else {
      clearInterval(interval);
      countdown.textContent = '📸';
      // Short flash then capture
      setTimeout(() => {
        captureAndEnroll();
        hideEnrollOverlay();
      }, 350);
    }
  }, 1000);
}

function hideEnrollOverlay() {
  document.getElementById('enrollOverlay').classList.remove('active');
  enrolling = false;
}

async function captureAndEnroll() {
  // Capture raw video frame (before blur is applied)
  const snap = document.createElement('canvas');
  snap.width  = videoEl.videoWidth;
  snap.height = videoEl.videoHeight;
  snap.getContext('2d').drawImage(videoEl, 0, 0);

  // Show snapshot preview immediately
  const previewBox = document.getElementById('enrollPreviewBox');
  const previewImg = document.getElementById('enrollPreview');
  const previewLbl = document.getElementById('enrollPreviewLabel');
  const idleBox    = document.getElementById('enrollIdle');

  previewImg.src = snap.toDataURL('image/jpeg', 0.92);
  previewBox.classList.add('visible');
  idleBox.style.display    = 'none';
  previewLbl.textContent   = 'Enrolling...';
  previewLbl.className     = 'enroll-preview-label';

  // POST to /enroll
  snap.toBlob(async (blob) => {
    const form = new FormData();
    form.append('file', blob, 'enroll.jpg');
    form.append('name', 'owner');

    try {
      const r = await fetch(`${API}/enroll`, { method: 'POST', body: form });
      const d = await r.json();

      if (r.ok) {
        previewLbl.textContent = "✅ Enrolled — your face won't be blurred";
        previewLbl.className   = 'enroll-preview-label success';
        toast('Face enrolled!', 'success');
        logDetection('face', null, '[ENROLL] Owner face registered — will not be blurred');
      } else {
        previewLbl.textContent = '❌ ' + (d.detail || 'No face detected — press E to retry');
        previewLbl.className   = 'enroll-preview-label error';
        toast('Enroll failed — try again', 'error');
      }
    } catch {
      previewLbl.textContent = '❌ Server unreachable';
      previewLbl.className   = 'enroll-preview-label error';
      toast('Server unreachable', 'error');
    }
  }, 'image/jpeg', 0.92);
}

function resetEnroll() {
  document.getElementById('enrollPreviewBox').classList.remove('visible');
  document.getElementById('enrollPreview').src = '';
  document.getElementById('enrollIdle').style.display = '';
  const sub = document.getElementById('enrollSub');
  sub.textContent = processing ? 'Camera live — press E to scan' : 'Start the camera first';
  sub.className   = processing ? 'enroll-sub ready' : 'enroll-sub';
}

// ── Processing loop ───────────────────────────────────────────────────────────
async function processLoop() {
  if (!processing) return;
  requestAnimationFrame(processLoop);
  if (pendingRequest) return;

  ctxCap.drawImage(videoEl, 0, 0, canCap.width, canCap.height);

  canCap.toBlob(async (blob) => {
    if (!blob || pendingRequest) return;
    pendingRequest = true;

    try {
      const form = new FormData();
      form.append('file', blob, 'frame.jpg');

      const resp = await fetch(`${API}/process_frame`, { method: 'POST', body: form });

      if (!resp.ok) {
        ctx.drawImage(videoEl, 0, 0);
        pendingRequest = false;
        return;
      }

      const imgBlob = await resp.blob();
      const url     = URL.createObjectURL(imgBlob);
      const img     = new Image();

      img.onload = () => {
        ctx.drawImage(img, 0, 0, canOut.width, canOut.height);
        URL.revokeObjectURL(url);

        const now = performance.now();
        const dt  = now - lastFrameTime;
        lastFrameTime = now;
        fpsSmooth = fpsSmooth * 0.8 + (1000 / dt) * 0.2;
        document.getElementById('fpsBadge').textContent = fpsSmooth.toFixed(1) + ' fps';
        document.getElementById('fpsLabel').textContent = fpsSmooth.toFixed(1) + ' fps';

        pendingRequest = false;
      };
      img.src = url;

      frameId++;

      const detectionHeader = resp.headers.get('X-Detections');
      if (detectionHeader) {
        try { handleDetections(JSON.parse(detectionHeader)); } catch {}
      }
    } catch {
      ctx.drawImage(videoEl, 0, 0);
      pendingRequest = false;
    }
  }, 'image/jpeg', 0.85);
}

// ── Detection log & stats ─────────────────────────────────────────────────────
function handleDetections(detections) {
  for (const d of detections) {
    logDetection(d.type, d.confidence);
    if (d.type === 'face')     { stats.faces++;  document.getElementById('statFaces').textContent  = stats.faces; }
    if (d.type === 'plate')    { stats.plates++; document.getElementById('statPlates').textContent = stats.plates; }
    if (d.type === 'text_pii') { stats.pii++;    document.getElementById('statPii').textContent    = stats.pii; }
    if (d.type === 'nsfw')     { stats.nsfw++;   document.getElementById('statNsfw').textContent   = stats.nsfw; }
  }
}

function logDetection(type, conf, customMsg = null) {
  const box     = document.getElementById('logBox');
  const time    = new Date().toTimeString().split(' ')[0];
  const clsMap  = { face: 'log-face', plate: 'log-plate', text_pii: 'log-pii', nsfw: 'log-nsfw', card: 'log-card', audio_pii: 'log-pii' };
  const cls     = clsMap[type] || '';
  const label   = type.replace('_', ' ').toUpperCase();
  const confStr = conf ? ` (${(conf * 100).toFixed(0)}%)` : '';
  const message = customMsg || `[${label}]${confStr} detected &amp; blurred`;

  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span class="log-time">${time}</span><span class="${cls}">${message}</span>`;
  box.prepend(el);
  while (box.children.length > 50) box.removeChild(box.lastChild);
}

// ── Config update ─────────────────────────────────────────────────────────────
async function updateConfig() {
  const body = {
    blur_faces:    document.getElementById('tog_blur_faces').checked,
    blur_plates:   document.getElementById('tog_blur_plates').checked,
    blur_cards:    document.getElementById('tog_blur_cards').checked,
    blur_nsfw:     document.getElementById('tog_blur_nsfw').checked,
    blur_text_pii: document.getElementById('tog_blur_text_pii').checked,
    blur_strength: parseInt(document.getElementById('blurStrength').value),
  };
  try {
    await fetch(`${API}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {}
}

// ── Snapshot ──────────────────────────────────────────────────────────────────
function takeSnapshot() {
  const link    = document.createElement('a');
  link.download = `blurberry_${Date.now()}.jpg`;
  link.href     = canOut.toDataURL('image/jpeg', 0.92);
  link.click();
  toast('Snapshot saved!', 'success');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, type = '') {
  const el       = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = type; }, 2800);
}