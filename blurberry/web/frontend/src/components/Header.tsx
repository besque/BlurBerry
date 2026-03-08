import { useState, useEffect } from 'react';
import { fetchStatus } from '../api';

export function Header() {
  const [status, setStatus] = useState<'checking' | 'ready' | 'loading' | 'error'>('checking');

  useEffect(() => {
    const check = async () => {
      try {
        const d = await fetchStatus();
        setStatus(d.pipeline_ready ? 'ready' : 'loading');
      } catch {
        setStatus('error');
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => clearInterval(t);
  }, []);

  const statusText =
    status === 'ready'
      ? 'pipeline ready'
      : status === 'error'
        ? 'server offline'
        : status === 'loading'
          ? 'server up / pipeline loading'
          : 'checking...';

  return (
    <header>
      <div className="logo">
        <div className="logo-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
            <circle cx="12" cy="12" r="9" fill="url(#blueberry-grad)" />
            <circle cx="9" cy="10" r="1.5" fill="rgba(255,255,255,0.4)" />
            <circle cx="15" cy="10" r="1.5" fill="rgba(255,255,255,0.4)" />
            <circle cx="12" cy="14" r="1" fill="rgba(255,255,255,0.3)" />
            <defs>
              <linearGradient id="blueberry-grad" x1="4" y1="4" x2="20" y2="20" gradientUnits="userSpaceOnUse">
                <stop stopColor="#4c1d95" />
                <stop offset="0.5" stopColor="#5b21b6" />
                <stop offset="1" stopColor="#3730a3" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        BlurBerry <span>AI</span>
      </div>
      <div className={`status-pill ${status === 'ready' ? 'ready' : ''} ${status === 'error' ? 'error' : ''}`}>
        <div className="status-dot" />
        <span>{statusText}</span>
      </div>
    </header>
  );
}
