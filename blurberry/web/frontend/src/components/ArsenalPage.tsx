import { useState } from 'react';
import './ArsenalPage.css';

export type ArsenalFeatureId = 'face' | 'card' | 'plate' | 'audio' | 'text_pii' | 'nsfw';

export interface ArsenalFeature {
  id: ArsenalFeatureId;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const FEATURES: ArsenalFeature[] = [
  {
    id: 'face',
    title: 'Face Blurring',
    description: 'Automatically blur faces in real-time.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M8 14s1.5 2 4 2 4-2 4-2" />
        <line x1="9" y1="9" x2="9.01" y2="9" />
        <line x1="15" y1="9" x2="15.01" y2="9" />
      </svg>
    ),
  },
  {
    id: 'card',
    title: 'Credit Cards',
    description: 'Detect and blur credit card numbers.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="5" width="20" height="14" rx="2" />
        <line x1="2" y1="10" x2="22" y2="10" />
      </svg>
    ),
  },
  {
    id: 'plate',
    title: 'Number Plates',
    description: 'Blur license plates and vehicle IDs.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 18V8a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2" />
        <path d="M20 18V8a2 2 0 0 0-2-2h-2" />
        <path d="M18 12h.01" />
      </svg>
    ),
  },
  {
    id: 'audio',
    title: 'Audio Blurring',
    description: 'Filter and redact sensitive speech in audio.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="22" />
      </svg>
    ),
  },
  {
    id: 'text_pii',
    title: 'Text PII',
    description: 'Detect and hide sensitive text (IDs, names).',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
    title: 'NSFW Detection',
    description: 'Blur inappropriate visual content.',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
  },
];

interface ArsenalPageProps {
  onNext: (selected: ArsenalFeatureId[]) => void;
}

export function ArsenalPage({ onNext }: ArsenalPageProps) {
  const [selected, setSelected] = useState<Set<ArsenalFeatureId>>(new Set(['face', 'card', 'plate']));

  const toggle = (id: ArsenalFeatureId) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="arsenal-page">
      <h1 className="arsenal-title">Choose your arsenal.</h1>
      <p className="arsenal-subtitle">Select the AI protection tools you need for your streams.</p>

      <div className="arsenal-grid">
        {FEATURES.map((f) => (
          <button
            key={f.id}
            type="button"
            className={`arsenal-card ${selected.has(f.id) ? 'selected' : ''}`}
            onClick={() => toggle(f.id)}
          >
            <span className="arsenal-card-icon">{f.icon}</span>
            <div className="arsenal-card-text">
              <span className="arsenal-card-title">{f.title}</span>
              <span className="arsenal-card-desc">{f.description}</span>
            </div>
            {selected.has(f.id) && <span className="arsenal-card-dot" />}
          </button>
        ))}
      </div>

      <div className="arsenal-nav">
        <span className="arsenal-nav-back-placeholder" />
        <div className="arsenal-dots">
          <span className="arsenal-dot active" />
          <span className="arsenal-dot" />
          <span className="arsenal-dot" />
        </div>
        <button type="button" className="arsenal-next" onClick={() => onNext(Array.from(selected))}>
          Next
        </button>
      </div>
    </div>
  );
}
