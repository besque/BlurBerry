import { useState, useRef, useEffect, useCallback } from 'react';
import { fetchEnrollDetect, fetchEnroll } from '../api';
import './FaceCaptureModal.css';

const FRAMES_NEEDED = 15;
const CAPTURE_INTERVAL_MS = 200;
const DETECT_INTERVAL_MS = 150;

interface FaceCaptureModalProps {
  onClose: () => void;
  onCaptureComplete: () => void;
  onSkipAndFinish?: () => void;
}

export function FaceCaptureModal({ onClose, onCaptureComplete }: FaceCaptureModalProps) {
  const [name, setName] = useState('');
  const [step, setStep] = useState<'idle' | 'capturing' | 'done'>('idle');
  const [collectedCount, setCollectedCount] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasOverlayRef = useRef<HTMLCanvasElement>(null);
  const previewWrapRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const detectIntervalRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const framesRef = useRef<Blob[]>([]);
  const [enrollError, setEnrollError] = useState<string | null>(null);
  const [enrolling, setEnrolling] = useState(false);

  useEffect(() => {
    let s: MediaStream | null = null;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((st) => {
        s = st;
        setStream(st);
        if (videoRef.current) {
          videoRef.current.srcObject = st;
          videoRef.current.play().catch(() => {});
        }
      })
      .catch(() => {});
    return () => {
      if (s) s.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    if (!stream || !videoRef.current) return;
    videoRef.current.srcObject = stream;
    videoRef.current.play().catch(() => {});
  }, [stream]);

  const captureFrame = useCallback((): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return Promise.resolve(null);
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d')!.drawImage(video, 0, 0);
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92);
    });
  }, []);

  const drawFaceBox = useCallback((bbox: number[], videoW: number, videoH: number) => {
    const wrap = previewWrapRef.current;
    const canvas = canvasOverlayRef.current;
    const video = videoRef.current;
    if (!wrap || !canvas || !video) return;
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    const scale = Math.min(wrapW / videoW, wrapH / videoH);
    const drawW = videoW * scale;
    const drawH = videoH * scale;
    const offsetX = (wrapW - drawW) / 2;
    const offsetY = (wrapH - drawH) / 2;
    const [x1, y1, x2, y2] = bbox;
    const rx1 = offsetX + x1 * scale;
    const ry1 = offsetY + y1 * scale;
    const rw = (x2 - x1) * scale;
    const rh = (y2 - y1) * scale;
    canvas.width = wrapW;
    canvas.height = wrapH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, wrapW, wrapH);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 4;
    ctx.strokeRect(rx1, ry1, rw, rh);
  }, []);

  const runDetect = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;
    const blob = await captureFrame();
    if (!blob) return null;
    try {
      const form = new FormData();
      form.append('file', blob, 'detect.jpg');
      const r = await fetch(`${API}/enroll/detect`, { method: 'POST', body: form });
      const d = await r.json();
      const faces = (d && d.faces) || [];
      if (faces.length > 0) {
        const largest = faces.reduce((a: { bbox: number[] }, b: { bbox: number[] }) =>
          (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) >
          (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]) ? b : a
        );
        drawFaceBox(largest.bbox, video.videoWidth, video.videoHeight);
        setFaceDetected(true);
        return largest.bbox;
      }
      setFaceDetected(false);
      const c = canvasOverlayRef.current;
      if (c) {
        const w = c.clientWidth || 1;
        const h = c.clientHeight || 1;
        c.width = w;
        c.height = h;
        c.getContext('2d')?.clearRect(0, 0, w, h);
      }
      return null;
    } catch {
      return null;
    }
  }, [captureFrame, drawFaceBox]);

  useEffect(() => {
    if (step !== 'idle' && step !== 'capturing') return;
    const tick = () => { runDetect(); };
    tick();
    detectIntervalRef.current = setInterval(tick, DETECT_INTERVAL_MS);
    return () => {
      if (detectIntervalRef.current) clearInterval(detectIntervalRef.current);
    };
  }, [step, runDetect]);

  const startCapture = useCallback(() => {
    if (step === 'idle') {
      framesRef.current = [];
      setEnrollError(null);
      setStep('capturing');
      setCollectedCount(0);
    }
  }, [step]);

  useEffect(() => {
    if (step !== 'capturing') return;
    const tick = async () => {
      const blob = await captureFrame();
      if (!blob) return;
      const form = new FormData();
      form.append('file', blob, 'check.jpg');
      try {
        const d = await fetchEnrollDetect(form);
        const faces = d.faces || [];
        if (faces.length > 0) {
          framesRef.current.push(blob);
          setCollectedCount((c) => {
            const next = c + 1;
            if (next >= FRAMES_NEEDED) {
              if (intervalRef.current) clearInterval(intervalRef.current);
              setStep('done');
            }
            return next;
          });
        }
      } catch {
        // still push frame if detect fails (network) so we don't block
        framesRef.current.push(blob);
        setCollectedCount((c) => {
          const next = c + 1;
          if (next >= FRAMES_NEEDED) {
            if (intervalRef.current) clearInterval(intervalRef.current);
            setStep('done');
          }
          return next;
        });
      }
    };
    intervalRef.current = setInterval(tick, CAPTURE_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [step, captureFrame]);

  const handleStart = async () => {
    const blobs = framesRef.current;
    if (!blobs.length) {
      setEnrollError('No frames captured. Try again.');
      return;
    }
    setEnrolling(true);
    setEnrollError(null);
    const form = new FormData();
    blobs.forEach((blob, i) => form.append('files', blob, `enroll_${i}.jpg`));
    form.append('name', (name && name.trim()) || 'owner');
    try {
      const r = await fetchEnroll(form);
      const d = r.ok ? await r.json() : { detail: (await r.json().catch(() => ({}))).detail || 'Enroll failed' };
      if (r.ok) {
        if (stream) stream.getTracks().forEach((t) => t.stop());
        setStream(null);
        onCaptureComplete();
      } else {
        setEnrollError(typeof d.detail === 'string' ? d.detail : 'Enrollment failed');
      }
    } catch {
      setEnrollError('Server unreachable');
    } finally {
      setEnrolling(false);
    }
  };

  return (
    <div className="face-capture-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="face-capture-modal">
        <div className="face-capture-header">
          <span className="face-capture-header-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 7l-7 5 7 5V7z" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
          </span>
          <span className="face-capture-header-title">Register Your Face</span>
          <button type="button" className="face-capture-close" onClick={onClose} aria-label="Close">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="face-capture-body">
          <label className="face-capture-label">Your Name</label>
          <input
            type="text"
            className="face-capture-input"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={step === 'capturing'}
          />

          <div className="face-capture-preview-wrap" ref={previewWrapRef}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="face-capture-video"
            />
            <canvas
              ref={canvasOverlayRef}
              className="face-capture-canvas-overlay"
              aria-hidden
            />
            <div className="face-capture-preview-overlay">
              {step === 'idle' && 'Position your face in the frame — green box when detected'}
              {step === 'capturing' && (faceDetected ? 'Hold still…' : 'Face the camera')}
            </div>
            {step === 'capturing' && (
              <div className="face-capture-progress">
                <div className="face-capture-progress-bar" style={{ width: `${(collectedCount / FRAMES_NEEDED) * 100}%` }} />
                <span>{collectedCount} / {FRAMES_NEEDED} frames</span>
              </div>
            )}
          </div>

          <div className="face-capture-instructions">
            <div className="face-capture-instructions-title">Instructions:</div>
            <ul>
              <li>Make sure your face is clearly visible and well-lit</li>
              <li>Look directly at the camera</li>
              <li>We&apos;ll capture 15 frames in 3 seconds</li>
              <li>Remove glasses if possible for better accuracy</li>
            </ul>
          </div>

          {step === 'idle' && (
            <button type="button" className="face-capture-btn" onClick={startCapture}>
              Start Capture
            </button>
          )}
          {step === 'done' && (
            <>
              {enrollError && <p className="face-capture-error">{enrollError}</p>}
              <button
                type="button"
                className="face-capture-btn"
                onClick={handleStart}
                disabled={enrolling}
              >
                {enrolling ? 'Enrolling…' : 'Enroll & Start'}
              </button>
            </>
          )}
          {step === 'capturing' && (
            <button type="button" className="face-capture-btn face-capture-btn-disabled" disabled>
              Capturing…
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
