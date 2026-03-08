import { useState, useRef, useCallback, useEffect } from 'react';
import { fetchEnrollPreview, fetchEnroll } from '../api';
import './EnrollPage.css';

const FRAMES_NEEDED = 30;
const PREVIEW_FPS = 12;

interface EnrollPageProps {
  onStartStream: () => void;
}

type EnrollResult = 'idle' | 'success' | 'error';

export function EnrollPage({ onStartStream }: EnrollPageProps) {
  const [webcamAllowed, setWebcamAllowed] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [collecting, setCollecting] = useState(false);
  const [collectedCount, setCollectedCount] = useState(0);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<EnrollResult>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [backendNotReady, setBackendNotReady] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const framesRef = useRef<Blob[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const previewLoopRef = useRef<number | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const stopWebcam = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (previewLoopRef.current !== null) cancelAnimationFrame(previewLoopRef.current);
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    setPreviewUrl(null);
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
    }
    setWebcamAllowed(false);
    setCollecting(false);
    setCollectedCount(0);
    framesRef.current = [];
    setResult('idle');
    setErrorMessage('');
  }, [stream]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (previewLoopRef.current !== null) cancelAnimationFrame(previewLoopRef.current);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [stream]);

  useEffect(() => {
    if (!stream || !videoRef.current) return;
    videoRef.current.srcObject = stream;
    videoRef.current.play().catch(() => {});
  }, [stream]);

  const onAllowWebcam = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      setStream(s);
      setWebcamAllowed(true);
      setResult('idle');
      setErrorMessage('');
      framesRef.current = [];
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : 'Could not access webcam');
    }
  };

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

  const sendPreviewFrame = useCallback(async () => {
    const blob = await captureFrame();
    if (!blob || !webcamAllowed) return;

    const form = new FormData();
    form.append('file', blob, 'preview.jpg');
    form.append('collecting', collecting ? '1' : '0');
    form.append('count', String(collectedCount));
    form.append('frames_needed', String(FRAMES_NEEDED));

    try {
      const r = await fetchEnrollPreview(form);
      if (r.status === 503) {
        setBackendNotReady(true);
        return;
      }
      setBackendNotReady(false);
      if (!r.ok) return;
      const imageBlob = await r.blob();
      const url = URL.createObjectURL(imageBlob);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = url;
      setPreviewUrl(url);
    } catch {
      setBackendNotReady(true);
    }
  }, [webcamAllowed, collecting, collectedCount, captureFrame]);

  useEffect(() => {
    if (!webcamAllowed || !videoRef.current) return;

    let rafId: number;
    let last = 0;
    const interval = 1000 / PREVIEW_FPS;

    const loop = (t: number) => {
      rafId = requestAnimationFrame(loop);
      if (t - last >= interval) {
        last = t;
        sendPreviewFrame();
      }
    };
    rafId = requestAnimationFrame(loop);
    previewLoopRef.current = rafId;
    return () => {
      cancelAnimationFrame(rafId);
      previewLoopRef.current = null;
    };
  }, [webcamAllowed, sendPreviewFrame]);

  const onStartCollecting = useCallback(() => {
    if (!webcamAllowed || !videoRef.current) return;
    setCollecting(true);
    setCollectedCount(0);
    framesRef.current = [];
    setResult('idle');
    setErrorMessage('');
  }, [webcamAllowed]);

  useEffect(() => {
    if (!collecting || collectedCount >= FRAMES_NEEDED) return;

    const tick = async () => {
      const blob = await captureFrame();
      if (blob) {
        framesRef.current.push(blob);
        setCollectedCount(framesRef.current.length);
      }
    };

    intervalRef.current = setInterval(tick, 150);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [collecting, collectedCount, captureFrame]);

  useEffect(() => {
    if (collectedCount < FRAMES_NEEDED) return;
    setCollecting(false);
    if (intervalRef.current) clearInterval(intervalRef.current);

    const submit = async () => {
      setSaving(true);
      const form = new FormData();
      framesRef.current.forEach((b, i) => form.append('files', b, `enroll_${i}.jpg`));
      form.append('name', 'owner');

      try {
        const r = await fetchEnroll(form);
        if (r.status === 503) {
          setErrorMessage('Backend not ready. Start the server with: python main.py');
          setResult('error');
          setSaving(false);
          return;
        }
        const d = await r.json();
        if (r.ok) {
          setResult('success');
        } else {
          setResult('error');
          setErrorMessage(d.detail || 'No face detected in any image');
        }
      } catch {
        setResult('error');
        setErrorMessage('Server unreachable');
      }
      setSaving(false);
      framesRef.current = [];
    };

    submit();
  }, [collectedCount]);

  return (
    <div className="enroll-page">
      <div className="enroll-card">
        <h2 className="enroll-title">Enroll your face</h2>
        <p className="enroll-desc">Hold still and face the camera. We’ll collect {FRAMES_NEEDED} frames (like test_face.py).</p>

        {!webcamAllowed ? (
          <>
            <div className="enroll-preview-placeholder">
              <span className="enroll-preview-icon">📷</span>
              <p>Allow webcam to show your camera feed</p>
            </div>
            <button type="button" className="btn btn-primary btn-enroll-cta" onClick={onAllowWebcam}>
              Allow webcam
            </button>
            {errorMessage && <p className="enroll-error">{errorMessage}</p>}
          </>
        ) : (
          <>
            <div className="enroll-preview-wrap">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="enroll-video enroll-video-hidden"
              />
              {backendNotReady && (
                <div className="enroll-preview-loading enroll-backend-error">
                  Pipeline not ready — start the backend with: <code>python main.py</code>
                </div>
              )}
              {!backendNotReady && previewUrl ? (
                <img src={previewUrl} alt="Live preview with face detection" className="enroll-preview-img" />
              ) : !backendNotReady ? (
                <div className="enroll-preview-loading">Connecting to server…</div>
              ) : null}
              {collecting && (
                <div className="enroll-progress-overlay">
                  <div className="enroll-progress-bar-wrap">
                    <div
                      className="enroll-progress-bar"
                      style={{ width: `${(collectedCount / FRAMES_NEEDED) * 100}%` }}
                    />
                  </div>
                  <p className="enroll-progress-text">Collecting: {collectedCount}/{FRAMES_NEEDED}</p>
                </div>
              )}
              {saving && (
                <div className="enroll-progress-overlay">
                  <p className="enroll-progress-text">Saving…</p>
                </div>
              )}
            </div>

            {result === 'idle' && !collecting && !saving && (
              <button
                type="button"
                className="btn btn-primary btn-enroll-cta"
                onClick={onStartCollecting}
              >
                Start collecting
              </button>
            )}

            {result === 'success' && (
              <div className="enroll-result enroll-result-success">
                <p>Face enrolled. You can start streaming.</p>
                <div className="enroll-result-actions">
                  <button type="button" className="btn btn-primary" onClick={onStartStream}>
                    Start stream
                  </button>
                </div>
              </div>
            )}

            {result === 'error' && (
              <div className="enroll-result enroll-result-error">
                <p>{errorMessage || 'Enrollment failed'}</p>
                <div className="enroll-result-actions">
                  <button type="button" className="btn btn-ghost" onClick={onStartCollecting}>
                    Try again
                  </button>
                  <button type="button" className="btn btn-primary" onClick={onStartStream}>
                    Skip and start stream
                  </button>
                </div>
              </div>
            )}

            {result === 'idle' && !collecting && !saving && (
              <button type="button" className="btn btn-ghost btn-enroll-back" onClick={stopWebcam}>
                Turn off webcam
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
