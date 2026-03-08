import type { ConfigState, StatsState } from '../hooks/useBlurBerry';

interface SidebarProps {
  stats: StatsState;
  config: ConfigState;
  enrollPreviewUrl: string | null;
  enrollLabel: string;
  enrollLabelSuccess: boolean;
  streaming: boolean;
  onConfigChange: (update: Partial<ConfigState>) => void;
  onResetEnroll: () => void;
}

export function Sidebar({
  stats,
  config,
  enrollPreviewUrl,
  enrollLabel,
  enrollLabelSuccess,
  streaming,
  onConfigChange,
  onResetEnroll,
}: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="section">
        <div className="section-title">Blur Modules</div>

        <div className="toggle-row">
          <span className="toggle-label">Faces <span className="tag">insightface</span></span>
          <label className="switch">
            <input type="checkbox" checked={config.blur_faces} onChange={(e) => onConfigChange({ blur_faces: e.target.checked })} />
            <span className="slider" />
          </label>
        </div>
        <div className="toggle-row">
          <span className="toggle-label">License Plates <span className="tag">yolo</span></span>
          <label className="switch">
            <input type="checkbox" checked={config.blur_plates} onChange={(e) => onConfigChange({ blur_plates: e.target.checked })} />
            <span className="slider" />
          </label>
        </div>
        <div className="toggle-row">
          <span className="toggle-label">ID / Credit Cards <span className="tag">yolo</span></span>
          <label className="switch">
            <input type="checkbox" checked={config.blur_cards} onChange={(e) => onConfigChange({ blur_cards: e.target.checked })} />
            <span className="slider" />
          </label>
        </div>
        <div className="toggle-row">
          <span className="toggle-label">NSFW Regions <span className="tag">nudenet</span></span>
          <label className="switch">
            <input type="checkbox" checked={config.blur_nsfw} onChange={(e) => onConfigChange({ blur_nsfw: e.target.checked })} />
            <span className="slider" />
          </label>
        </div>
        <div className="toggle-row">
          <span className="toggle-label">Text PII <span className="tag">paddleocr</span></span>
          <label className="switch">
            <input type="checkbox" checked={config.blur_text_pii} onChange={(e) => onConfigChange({ blur_text_pii: e.target.checked })} />
            <span className="slider" />
          </label>
        </div>

        <div className="range-row" style={{ marginTop: 14 }}>
          <div className="range-label">
            <span>Blur Strength</span>
            <span>{config.blur_strength}</span>
          </div>
          <input
            type="range"
            min={11}
            max={99}
            step={2}
            value={config.blur_strength}
            onChange={(e) => onConfigChange({ blur_strength: parseInt(e.target.value, 10) })}
          />
        </div>
      </div>

      <div className="section">
        <div className="section-title">Session Stats</div>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{stats.faces}</div>
            <div className="stat-label">Faces Blurred</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.plates}</div>
            <div className="stat-label">Plates</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.pii}</div>
            <div className="stat-label">Text PII</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.nsfw}</div>
            <div className="stat-label">NSFW</div>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-title">Owner Face</div>

        {enrollPreviewUrl ? (
          <div className="enroll-preview-box visible">
            <img src={enrollPreviewUrl} alt="Enrolled face" />
            <div className={`enroll-preview-label ${enrollLabelSuccess ? 'success' : enrollLabelSuccess === false && enrollLabel ? 'error' : ''}`}>
              {enrollLabel}
            </div>
            <button type="button" className="btn btn-ghost" style={{ width: '100%', marginTop: 8 }} onClick={onResetEnroll}>
              🔄 Re-enroll
            </button>
          </div>
        ) : (
          <div className="enroll-idle">
            <div className="enroll-key-hint">
              <span className="key-badge">E</span>
              <span>
                Press <strong>E</strong> while the camera
                <br />
                is live to scan your face
              </span>
            </div>
            <p className={`enroll-sub ${streaming ? 'ready' : ''}`}>{streaming ? 'Camera live — press E to scan' : 'Enroll from the start page first'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
