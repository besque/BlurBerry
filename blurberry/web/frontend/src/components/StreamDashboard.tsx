import { useState, useRef, useEffect } from 'react';
import { useBlurBerry } from '../hooks/useBlurBerry';
import { VideoPanel } from './VideoPanel';
import type { ArsenalFeatureId } from './ArsenalPage';
import './StreamDashboard.css';

function arsenalToFeatureSet(ids: ArsenalFeatureId[]) {
  return new Set(ids);
}

function dashboardConfigToBackend(features: Set<string>, blurStrength: number) {
  return {
    blur_faces: features.has('face'),
    blur_plates: features.has('plate'),
    blur_cards: features.has('card'),
    blur_nsfw: features.has('nsfw'),
    blur_text_pii: features.has('text_pii'),
    blur_strength: blurStrength,
  };
}

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface StreamDashboardProps {
  selectedArsenal: ArsenalFeatureId[];
}

export function StreamDashboard({ selectedArsenal }: StreamDashboardProps) {
  const blurberry = useBlurBerry();
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const sessionStartRef = useRef<number | null>(null);
  const features = arsenalToFeatureSet(selectedArsenal);

  useEffect(() => {
    blurberry.updateConfig(dashboardConfigToBackend(features, 51));
  }, [selectedArsenal.join(',')]);

  useEffect(() => {
    if (blurberry.streaming) {
      if (sessionStartRef.current === null) sessionStartRef.current = Date.now();
      const t = setInterval(() => {
        if (sessionStartRef.current !== null)
          setSessionSeconds(Math.floor((Date.now() - sessionStartRef.current) / 1000));
      }, 1000);
      return () => clearInterval(t);
    }
    sessionStartRef.current = null;
    setSessionSeconds(0);
  }, [blurberry.streaming]);

  return (
    <>
      <main className="stream-dashboard-simple">
        <div className="stream-simple-video-wrap">
          <VideoPanel
            videoRef={blurberry.videoRef}
            canvasOutputRef={blurberry.canvasOutputRef}
            canvasCaptureRef={blurberry.canvasCaptureRef}
            streaming={blurberry.streaming}
            fps={blurberry.fps}
            enrollCountdown={null}
            onStartCamera={blurberry.startCamera}
            onStopCamera={blurberry.stopCamera}
            onTakeSnapshot={blurberry.takeSnapshot}
            onStartEnrollCountdown={() => {}}
            hideEnroll
            serverStreamAvailable={blurberry.serverStreamAvailable}
            minimal
          />
        </div>
        <div className="stream-simple-bar">
          <span className="stream-simple-stat">
            {blurberry.streaming ? `${blurberry.fps.toFixed(1)} fps` : '—'}
          </span>
          <span className="stream-simple-stat">
            Time {formatTime(sessionSeconds)}
          </span>
          <span className="stream-simple-stat">
            Faces blurred {blurberry.stats.faces}
          </span>
        </div>
      </main>
      <div
        id="toast"
        className={`${blurberry.toast.message ? 'show' : ''} ${blurberry.toast.type}`}
        style={{ pointerEvents: 'none' }}
      >
        {blurberry.toast.message}
      </div>
    </>
  );
}
