import './DashboardSidebar.css';

export type FeatureId = 'face' | 'card' | 'plate' | 'audio' | 'text_pii' | 'nsfw';

interface FeatureDef {
  id: FeatureId;
  label: string;
  icon: React.ReactNode;
}

const FEATURES: FeatureDef[] = [
  {
    id: 'face',
    label: 'Face Blur',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M8 14s1.5 2 4 2 4-2 4-2" />
        <line x1="9" y1="9" x2="9.01" y2="9" />
        <line x1="15" y1="9" x2="15.01" y2="9" />
      </svg>
    ),
  },
  {
    id: 'card',
    label: 'Credit Card',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="5" width="20" height="14" rx="2" />
        <line x1="2" y1="10" x2="22" y2="10" />
      </svg>
    ),
  },
  {
    id: 'plate',
    label: 'Number Plates',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 18V8a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2" />
        <path d="M20 18V8a2 2 0 0 0-2-2h-2" />
        <path d="M18 12h.01" />
      </svg>
    ),
  },
  {
    id: 'audio',
    label: 'Audio Blur',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="22" />
      </svg>
    ),
  },
  {
    id: 'text_pii',
    label: 'Text PII',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <path d="M14 2v6h6" />
        <path d="M16 13H8" />
        <path d="M16 17H8" />
        <path d="M10 9H8" />
      </svg>
    ),
  },
  {
    id: 'nsfw',
    label: 'NSFW',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
  },
];

export interface DashboardConfig {
  features: Set<FeatureId>;
  blurStrength: number;
}

interface DashboardSidebarProps {
  config: DashboardConfig;
  onChange: (config: DashboardConfig) => void;
}

export function DashboardSidebar({ config, onChange }: DashboardSidebarProps) {
  const toggleFeature = (id: FeatureId) => {
    const next = new Set(config.features);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onChange({ ...config, features: next });
  };

  const setBlurStrength = (v: number) => {
    onChange({ ...config, blurStrength: v });
  };

  return (
    <div className="dashboard-sidebar">
      <div className="dashboard-sidebar-section">
        <div className="dashboard-sidebar-title">Privacy Controls</div>
        <div className="dashboard-sidebar-toggles">
          {FEATURES.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`dashboard-toggle-btn ${config.features.has(f.id) ? 'active' : ''}`}
              onClick={() => toggleFeature(f.id)}
              title={f.label}
            >
              <span className="dashboard-toggle-icon">{f.icon}</span>
              <span className="dashboard-toggle-label">{f.label}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="dashboard-sidebar-section">
        <div className="dashboard-sidebar-range">
          <span className="dashboard-range-label">Blur effect</span>
          <span className="dashboard-range-value">{config.blurStrength}</span>
        </div>
        <input
          type="range"
          min={11}
          max={99}
          step={2}
          value={config.blurStrength}
          onChange={(e) => setBlurStrength(parseInt(e.target.value, 10))}
          className="dashboard-blur-slider"
        />
      </div>
    </div>
  );
}
