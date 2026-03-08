import { useEffect } from 'react';
import type { RefObject } from 'react';

interface VideoPanelProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasOutputRef: RefObject<HTMLCanvasElement | null>;
  canvasCaptureRef: RefObject<HTMLCanvasElement | null>;
  streaming: boolean;
  fps: number;
  enrollCountdown: number | null;
  onStartCamera: () => void;
  onStopCamera: () => void;
  onTakeSnapshot: () => void;
  onStartEnrollCountdown: () => void;
  /** When true, hide enroll-from-camera UI (enrollment already done in wizard). */
  hideEnroll?: boolean;
  /** When true, start button will show server camera (virtual camera output). */
  serverStreamAvailable?: boolean;
  /** Minimal UI: only stream + Start/Stop/Snapshot, no enroll. */
  minimal?: boolean;
}

export function VideoPanel({
  videoRef,
  canvasOutputRef,
  canvasCaptureRef,
  streaming,
  fps,
  enrollCountdown,
  onStartCamera,
  onStopCamera,
  onTakeSnapshot,
  onStartEnrollCountdown,
  hideEnroll = false,
  serverStreamAvailable = false,
  minimal = false,
}: VideoPanelProps) {
  useEffect(() => {
    if (hideEnroll || minimal) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'e' || e.key === 'E') {
        if (streaming && enrollCountdown === null) onStartEnrollCountdown();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hideEnroll, minimal, streaming, enrollCountdown, onStartEnrollCountdown]);

  const label = serverStreamAvailable ? 'Start stream' : 'Start camera';

  return (
    <div className={`video-panel ${minimal ? 'video-panel-minimal' : ''}`}>
      <div className="video-toolbar">
        <button type="button" className="btn btn-primary" onClick={onStartCamera} style={{ display: streaming ? 'none' : '' }}>
          ▶ {label}
        </button>
        <button type="button" className="btn btn-danger" onClick={onStopCamera} style={{ display: streaming ? '' : 'none' }}>
          ■ Stop
        </button>
        <button type="button" className="btn btn-ghost" onClick={onTakeSnapshot} style={{ display: streaming ? '' : 'none' }}>
          📷 Snapshot
        </button>
      </div>

      <div className="video-container">
        {/* Video must be in layout (not display:none) so the browser decodes frames; we hide it off-screen */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video-source-hidden"
        />
        {streaming && (
          <div className="video-preview-wrap">
            <div className="video-preview-wrap-inner">
              <canvas id="canvasOutput" ref={canvasOutputRef} />
            </div>
          </div>
        )}
        <canvas ref={canvasCaptureRef} style={{ display: 'none' }} />

        {!streaming && (
          <div className="video-placeholder">
            <div className="icon">🫐</div>
            <p className="video-placeholder-main">{minimal ? 'Stopped' : 'Camera stopped'}</p>
            <p className="video-placeholder-sub">Click &quot;{label}&quot; to begin</p>
          </div>
        )}

        {!hideEnroll && !minimal && (
          <div className={`enroll-overlay ${enrollCountdown !== null ? 'active' : ''}`}>
            <div className="enroll-ring" />
            <div className="enroll-countdown">
              {enrollCountdown !== null ? (enrollCountdown > 0 ? enrollCountdown : '📸') : ''}
            </div>
            <div className="enroll-overlay-label">Hold still…</div>
          </div>
        )}

        {!minimal && (
          <div className="fps-badge" style={{ display: streaming ? '' : 'none' }}>
            {fps.toFixed(1)} fps
          </div>
        )}
      </div>

      {!hideEnroll && !minimal && (
        <div className="enroll-hint" style={{ display: streaming ? '' : 'none' }}>
          Press <kbd>E</kbd> to enroll your face — your face won&apos;t be blurred
        </div>
      )}
    </div>
  );
}
