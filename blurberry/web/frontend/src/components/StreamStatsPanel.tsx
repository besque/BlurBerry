import './StreamStatsPanel.css';

interface StreamStatsPanelProps {
  fps: number;
  streaming: boolean;
  facesBlurred?: number;
  /** Session duration in seconds (for timecode display) */
  sessionSeconds?: number;
  /** Process CPU % from server (null if unavailable) */
  cpuUsage?: number | null;
  /** Upload bitrate in bytes per second (frame size × FPS) */
  bitrateBps?: number;
}

function formatTimecode(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((n) => n.toString().padStart(2, '0')).join(':');
}

function formatBitrate(bps: number): string {
  if (bps >= 1024 * 1024) return `${(bps / (1024 * 1024)).toFixed(2)} MB/s`;
  if (bps >= 1024) return `${(bps / 1024).toFixed(1)} KB/s`;
  return `${bps} B/s`;
}

export function StreamStatsPanel({
  fps,
  streaming,
  facesBlurred = 0,
  sessionSeconds = 0,
  cpuUsage = null,
  bitrateBps = 0,
}: StreamStatsPanelProps) {
  return (
    <div className="stream-stats-panel">
      <div className="stream-stats-title">
        <span className="stream-stats-title-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </span>
        Stream Analytics
      </div>
      <p className="stream-stats-subtitle">Real-time performance metrics.</p>

      <div className="stream-stats-list">
        <div className="stream-stat-card stream-stat-card-fps">
          <span className="stream-stat-icon stream-stat-icon-fps">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </span>
          <div className="stream-stat-value">{fps.toFixed(1)}</div>
          <div className="stream-stat-label">FPS</div>
        </div>

        <div className="stream-stat-card">
          <span className="stream-stat-icon stream-stat-icon-cpu">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
              <rect x="9" y="9" width="6" height="6" />
              <line x1="9" y1="1" x2="9" y2="4" />
              <line x1="15" y1="1" x2="15" y2="4" />
              <line x1="9" y1="20" x2="9" y2="23" />
              <line x1="15" y1="20" x2="15" y2="23" />
              <line x1="20" y1="9" x2="23" y2="9" />
              <line x1="20" y1="14" x2="23" y2="14" />
              <line x1="1" y1="9" x2="4" y2="9" />
              <line x1="1" y1="14" x2="4" y2="14" />
            </svg>
          </span>
          <div className="stream-stat-value">{cpuUsage != null ? `${cpuUsage}%` : '—'}</div>
          <div className="stream-stat-label">CPU Usage</div>
        </div>

        <div className="stream-stat-card">
          <span className="stream-stat-icon stream-stat-icon-bitrate">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12.55a11 11 0 0 1 14.08 0" />
              <path d="M1.42 9a16 16 0 0 1 21.16 0" />
              <path d="M8.53 16.11a6 6 0 0 1 6.95 0" />
              <line x1="12" y1="20" x2="12.01" y2="20" />
            </svg>
          </span>
          <div className="stream-stat-value">{streaming && bitrateBps > 0 ? formatBitrate(bitrateBps) : '—'}</div>
          <div className="stream-stat-label">Bitrate</div>
        </div>

        <div className="stream-stats-status-box">
          <div className="stream-stats-status-title">Stream Status</div>
          <div className="stream-stats-status-row">
            <div className="stream-stats-status-item">
              <span className="stream-stats-status-label">Status</span>
              <span className={`stream-stats-status-value ${streaming ? 'live' : 'offline'}`}>
                {streaming ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
            <div className="stream-stats-status-item">
              <span className="stream-stats-status-label">Timecode</span>
              <span className="stream-stats-status-value">{formatTimecode(sessionSeconds)}</span>
            </div>
          </div>
        </div>

        <div className="stream-stat-card">
          <span className="stream-stat-icon stream-stat-icon-fps">👤</span>
          <div className="stream-stat-value">{facesBlurred}</div>
          <div className="stream-stat-label">Faces Blurred</div>
        </div>
      </div>
    </div>
  );
}
