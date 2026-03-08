import { useState, useRef, useCallback, useEffect } from 'react';
import {
  fetchStatus,
  fetchConfigUpdate,
  fetchProcessFrame,
  fetchEnroll,
  fetchAudioEvents,
  fetchStreamLatest,
  fetchStreamStats,
} from '../api';

export interface StatusState {
  pipelineReady: boolean;
  serverOnline: boolean;
}

export interface ConfigState {
  blur_faces: boolean;
  blur_plates: boolean;
  blur_cards: boolean;
  blur_nsfw: boolean;
  blur_text_pii: boolean;
  blur_strength: number;
}

export interface StatsState {
  faces: number;
  plates: number;
  pii: number;
  nsfw: number;
}

export interface LogEntry {
  id: number;
  time: string;
  type: string;
  message: string;
}

export function useBlurBerry() {
  const [status, setStatus] = useState<StatusState>({ pipelineReady: false, serverOnline: false });
  const [config, setConfigState] = useState<ConfigState>({
    blur_faces: true,
    blur_plates: true,
    blur_cards: true,
    blur_nsfw: true,
    blur_text_pii: true,
    blur_strength: 51,
  });
  const [stats, setStats] = useState<StatsState>({ faces: 0, plates: 0, pii: 0, nsfw: 0 });
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [toastState, setToastState] = useState<{ message: string; type: 'success' | 'error' | '' }>({ message: '', type: '' });
  const [fps, setFps] = useState(0);
  const [cpuUsage, setCpuUsage] = useState<number | null>(null);
  const [bitrateBps, setBitrateBps] = useState<number>(0);
  const [enrolled, setEnrolled] = useState(false);
  const [enrollPreviewUrl, setEnrollPreviewUrl] = useState<string | null>(null);
  const [enrollLabel, setEnrollLabel] = useState('');
  const [enrollLabelSuccess, setEnrollLabelSuccess] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [enrollCountdown, setEnrollCountdown] = useState<number | null>(null);
  const [serverStreamAvailable, setServerStreamAvailable] = useState(false);
  const [useServerStream, setUseServerStream] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const processingRef = useRef(false);
  const pendingRequestRef = useRef(false);
  const serverStreamRef = useRef(false);
  const serverStreamLastFetchRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  const fpsSmoothRef = useRef(0);
  const lastFrameBytesRef = useRef(0);
  const logIdRef = useRef(0);
  const displayModeRef = useRef<'video' | 'processed'>('video');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasOutputRef = useRef<HTMLCanvasElement>(null);
  const canvasCaptureRef = useRef<HTMLCanvasElement>(null);

  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const showToast = useCallback((message: string, type: 'success' | 'error' | '' = '') => {
    setToastState({ message, type });
    clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToastState({ message: '', type: '' }), 2800);
  }, []);

  const addLog = useCallback((type: string, message: string) => {
    const time = new Date().toTimeString().split(' ')[0];
    setLogEntries((prev) => [{ id: ++logIdRef.current, time, type, message }, ...prev].slice(0, 50));
  }, []);

  const checkStatus = useCallback(async () => {
    try {
      const d = await fetchStatus();
      setStatus({ pipelineReady: !!d.pipeline_ready, serverOnline: true });
      setServerStreamAvailable(!!d.server_stream_available);
      if (d.config) {
        setConfigState((c) => ({
          ...c,
          blur_faces: d.config.blur_faces ?? c.blur_faces,
          blur_plates: d.config.blur_plates ?? c.blur_plates,
          blur_cards: d.config.blur_cards ?? c.blur_cards,
          blur_nsfw: d.config.blur_nsfw ?? c.blur_nsfw,
          blur_text_pii: d.config.blur_text_pii ?? c.blur_text_pii,
          blur_strength: d.config.blur_strength ?? c.blur_strength,
        }));
      }
    } catch {
      setStatus((s) => ({ ...s, serverOnline: false }));
      setServerStreamAvailable(false);
    }
  }, []);

  useEffect(() => {
    checkStatus();
    const interval = streaming ? 2000 : 5000;
    const t = setInterval(checkStatus, interval);
    return () => clearInterval(t);
  }, [checkStatus, streaming]);

  useEffect(() => {
    const poll = async () => {
      try {
        const d = await fetchAudioEvents();
        for (const ev of d.events || []) {
          const types = (ev.findings || []).map((f: unknown[]) => (Array.isArray(f) ? f[1] : '')).join(', ');
          addLog('audio_pii', `🔴 Audio PII [${types}]: "${ev.text}"`);
        }
      } catch {}
    };
    const t = setInterval(poll, 500);
    return () => clearInterval(t);
  }, [addLog]);

  useEffect(() => {
    if (!streaming || !useServerStream) return;
    const pollStats = async () => {
      try {
        const s = await fetchStreamStats();
        setStats({ faces: s.faces, plates: s.plates, pii: s.pii, nsfw: s.nsfw });
      } catch {}
    };
    pollStats();
    const t = setInterval(pollStats, 400);
    return () => clearInterval(t);
  }, [streaming, useServerStream]);

  const updateConfig = useCallback(
    async (update: Partial<ConfigState>) => {
      const next = { ...config, ...update };
      setConfigState(next);
      try {
        await fetchConfigUpdate(next);
      } catch {}
    },
    [config]
  );

  const previewLoop = useCallback(() => {
    if (!processingRef.current || serverStreamRef.current) return;
    const video = videoRef.current;
    const canOut = canvasOutputRef.current;
    if (!video || !canOut || displayModeRef.current !== 'video') {
      requestAnimationFrame(previewLoop);
      return;
    }
    const ctx = canOut.getContext('2d');
    if (!ctx || video.readyState < 2) {
      requestAnimationFrame(previewLoop);
      return;
    }
    if (canOut.width !== video.videoWidth || canOut.height !== video.videoHeight) {
      canOut.width = video.videoWidth;
      canOut.height = video.videoHeight;
    }
    if (canOut.width > 0 && canOut.height > 0) {
      ctx.drawImage(video, 0, 0, canOut.width, canOut.height);
    }
    requestAnimationFrame(previewLoop);
  }, []);

  const serverStreamLoop = useCallback(() => {
    if (!processingRef.current || !serverStreamRef.current) return;
    const canOut = canvasOutputRef.current;
    if (!canOut) {
      requestAnimationFrame(serverStreamLoop);
      return;
    }
    const now = performance.now();
    const targetFps = 30;
    const minIntervalMs = 1000 / targetFps;
    if (now - serverStreamLastFetchRef.current < minIntervalMs) {
      requestAnimationFrame(serverStreamLoop);
      return;
    }
    serverStreamLastFetchRef.current = now;
    fetchStreamLatest()
      .then((r) => {
        if (!r.ok || r.status === 204) {
          requestAnimationFrame(serverStreamLoop);
          return;
        }
        return r.blob();
      })
      .then((blob) => {
        if (!blob || !serverStreamRef.current) {
          requestAnimationFrame(serverStreamLoop);
          return;
        }
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          const ctx = canOut.getContext('2d');
          if (ctx && serverStreamRef.current) {
            if (canOut.width !== img.width || canOut.height !== img.height) {
              canOut.width = img.width;
              canOut.height = img.height;
            }
            ctx.drawImage(img, 0, 0);
            const t = performance.now();
            const dt = t - lastFrameTimeRef.current;
            lastFrameTimeRef.current = t;
            fpsSmoothRef.current = fpsSmoothRef.current * 0.8 + (1000 / Math.max(dt, 1)) * 0.2;
            setFps(fpsSmoothRef.current);
          }
          URL.revokeObjectURL(url);
          requestAnimationFrame(serverStreamLoop);
        };
        img.onerror = () => {
          URL.revokeObjectURL(url);
          requestAnimationFrame(serverStreamLoop);
        };
        img.src = url;
      })
      .catch(() => requestAnimationFrame(serverStreamLoop));
  }, []);

  const processLoop = useCallback(() => {
    if (serverStreamRef.current) return;
    const video = videoRef.current;
    const canOut = canvasOutputRef.current;
    const canCap = canvasCaptureRef.current;
    if (!processingRef.current || !video || !canOut || !canCap || pendingRequestRef.current) {
      return;
    }
    const ctx = canOut.getContext('2d');
    const ctxCap = canCap.getContext('2d');
    if (!ctx || !ctxCap) {
      requestAnimationFrame(processLoop);
      return;
    }
    if (video.readyState < 2) {
      requestAnimationFrame(processLoop);
      return;
    }
    if (canOut.width !== video.videoWidth || canCap.width !== video.videoWidth) {
      canOut.width = canCap.width = video.videoWidth;
      canOut.height = canCap.height = video.videoHeight;
    }
    displayModeRef.current = 'video';
    ctxCap.drawImage(video, 0, 0, canCap.width, canCap.height);
    canCap.toBlob(
      (blob) => {
        if (!blob || pendingRequestRef.current) {
          requestAnimationFrame(processLoop);
          return;
        }
        pendingRequestRef.current = true;
        lastFrameBytesRef.current = blob.size;
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        fetchProcessFrame(form)
          .then((resp) => {
            if (!resp.ok) {
              displayModeRef.current = 'video';
              pendingRequestRef.current = false;
              requestAnimationFrame(processLoop);
              return;
            }
            return resp.blob();
          })
          .then((imgBlob) => {
            if (!imgBlob) {
              pendingRequestRef.current = false;
              requestAnimationFrame(processLoop);
              return;
            }
            const url = URL.createObjectURL(imgBlob);
            const img = new Image();
            img.onload = () => {
              ctx.drawImage(img, 0, 0, canOut.width, canOut.height);
              URL.revokeObjectURL(url);
              displayModeRef.current = 'processed';
              const now = performance.now();
              const dt = now - lastFrameTimeRef.current;
              lastFrameTimeRef.current = now;
              fpsSmoothRef.current = fpsSmoothRef.current * 0.8 + (1000 / dt) * 0.2;
              setFps(fpsSmoothRef.current);
              setBitrateBps(Math.round(lastFrameBytesRef.current * fpsSmoothRef.current));
              pendingRequestRef.current = false;
              requestAnimationFrame(processLoop);
            };
            img.onerror = () => {
              URL.revokeObjectURL(url);
              displayModeRef.current = 'video';
              pendingRequestRef.current = false;
              requestAnimationFrame(processLoop);
            };
            img.src = url;
          })
          .catch(() => {
            displayModeRef.current = 'video';
            pendingRequestRef.current = false;
            requestAnimationFrame(processLoop);
          });
      },
      'image/jpeg',
      0.85
    );
  }, []);

  const startCamera = useCallback(async () => {
    if (serverStreamAvailable) {
      serverStreamRef.current = true;
      setUseServerStream(true);
      processingRef.current = true;
      setStreaming(true);
      lastFrameTimeRef.current = performance.now();
      fpsSmoothRef.current = 0;
      requestAnimationFrame(serverStreamLoop);
      addLog('system', 'Server camera stream — showing virtual camera output');
      showToast('Server camera stream started', 'success');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
        });
        await video.play();
        await new Promise<void>((resolve) => {
          if (video.readyState >= 2) return resolve();
          video.addEventListener('canplay', () => resolve(), { once: true });
        });
        const canOut = canvasOutputRef.current;
        const canCap = canvasCaptureRef.current;
        if (canOut && canCap) {
          canOut.width = canCap.width = video.videoWidth;
          canOut.height = canCap.height = video.videoHeight;
        }
      }
      processingRef.current = true;
      setStreaming(true);
      displayModeRef.current = 'video';
      lastFrameTimeRef.current = performance.now();
      fpsSmoothRef.current = 0;
      requestAnimationFrame(previewLoop);
      requestAnimationFrame(processLoop);
      addLog('system', 'Camera started — preview active');
      showToast('Camera started', 'success');
    } catch (e) {
      showToast('Camera access denied: ' + (e instanceof Error ? e.message : String(e)), 'error');
    }
  }, [processLoop, previewLoop, serverStreamLoop, serverStreamAvailable, showToast, addLog]);

  const stopCamera = useCallback(() => {
    const wasServerStream = serverStreamRef.current;
    serverStreamRef.current = false;
    setUseServerStream(false);
    processingRef.current = false;
    setStreaming(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const canOut = canvasOutputRef.current;
    if (canOut) {
      const ctx = canOut.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canOut.width, canOut.height);
    }
    setFps(0);
    setBitrateBps(0);
    addLog('system', wasServerStream ? 'Server camera stopped' : 'Camera stopped');
    showToast('Camera stopped');
  }, [showToast, addLog]);

  const captureAndEnroll = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    const snap = document.createElement('canvas');
    snap.width = video.videoWidth;
    snap.height = video.videoHeight;
    snap.getContext('2d')!.drawImage(video, 0, 0);
    const dataUrl = snap.toDataURL('image/jpeg', 0.92);
    setEnrollPreviewUrl(dataUrl);
    setEnrollLabel('Enrolling...');
    setEnrollLabelSuccess(false);

    snap.toBlob(
      async (blob) => {
        if (!blob) return;
        const form = new FormData();
        form.append('files', blob, 'enroll.jpg');
        form.append('name', 'owner');
        try {
          const r = await fetchEnroll(form);
          const d = await r.json();
          if (r.ok) {
            setEnrollLabel("✅ Enrolled — your face won't be blurred");
            setEnrollLabelSuccess(true);
            setEnrolled(true);
            showToast('Face enrolled!', 'success');
            addLog('face', '[ENROLL] Owner face registered — will not be blurred');
          } else {
            setEnrollLabel('❌ ' + (d.detail || 'No face detected — press E to retry'));
            setEnrollLabelSuccess(false);
            showToast('Enroll failed — try again', 'error');
          }
        } catch {
          setEnrollLabel('❌ Server unreachable');
          setEnrollLabelSuccess(false);
          showToast('Server unreachable', 'error');
        }
      },
      'image/jpeg',
      0.92
    );
  }, [showToast, addLog]);

  const resetEnroll = useCallback(() => {
    setEnrollPreviewUrl(null);
    setEnrollLabel('');
    setEnrolled(false);
  }, []);

  useEffect(() => {
    if (enrollCountdown === null) return;
    if (enrollCountdown === 0) {
      captureAndEnroll();
      setEnrollCountdown(null);
      return;
    }
    const t = setTimeout(() => setEnrollCountdown((c) => (c !== null ? c - 1 : 0)), 1000);
    return () => clearTimeout(t);
  }, [enrollCountdown, captureAndEnroll]);

  const startEnrollCountdown = useCallback(() => {
    if (!streaming) return;
    setEnrollCountdown(3);
  }, [streaming]);

  const clearEnrollCountdown = useCallback(() => {
    setEnrollCountdown(null);
  }, []);

  const handleDetections = useCallback(
    (detections: { type: string; confidence?: number }[]) => {
      for (const d of detections) {
        const label = d.type.replace('_', ' ').toUpperCase();
        const confStr = d.confidence ? ` (${(d.confidence * 100).toFixed(0)}%)` : '';
        addLog(d.type, `[${label}]${confStr} detected & blurred`);
        setStats((s) => {
          const next = { ...s };
          if (d.type === 'face') next.faces++;
          if (d.type === 'plate') next.plates++;
          if (d.type === 'text_pii') next.pii++;
          if (d.type === 'nsfw') next.nsfw++;
          return next;
        });
      }
    },
    [addLog]
  );

  const takeSnapshot = useCallback(() => {
    const canOut = canvasOutputRef.current;
    if (!canOut) return;
    const link = document.createElement('a');
    link.download = `blurberry_${Date.now()}.jpg`;
    link.href = canOut.toDataURL('image/jpeg', 0.92);
    link.click();
    showToast('Snapshot saved!', 'success');
  }, [showToast]);

  return {
    status,
    config,
    setConfigState: updateConfig,
    stats,
    logEntries,
    toast: toastState,
    fps,
    cpuUsage,
    bitrateBps,
    streaming,
    serverStreamAvailable,
    useServerStream,
    enrolled,
    enrollPreviewUrl,
    enrollLabel,
    enrollLabelSuccess,
    videoRef,
    canvasOutputRef,
    canvasCaptureRef,
    streamRef,
    processingRef,
    checkStatus,
    startCamera,
    stopCamera,
    captureAndEnroll,
    resetEnroll,
    enrollCountdown,
    setEnrollCountdown,
    startEnrollCountdown,
    clearEnrollCountdown,
    handleDetections,
    updateConfig,
    takeSnapshot,
    showToast,
    addLog,
  };
}
