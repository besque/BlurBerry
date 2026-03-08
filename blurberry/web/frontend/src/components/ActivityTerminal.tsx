import { useRef, useEffect } from 'react';
import './ActivityTerminal.css';

export interface ActivityLine {
  id: string;
  time: string;
  type: 'info' | 'face' | 'plate' | 'card' | 'audio' | 'text_pii' | 'nsfw' | 'system';
  message: string;
}

interface ActivityTerminalProps {
  lines: ActivityLine[];
}

const TYPE_CLASS: Record<ActivityLine['type'], string> = {
  info: 'log-info',
  face: 'log-face',
  plate: 'log-plate',
  card: 'log-card',
  audio: 'log-audio',
  text_pii: 'log-pii',
  nsfw: 'log-nsfw',
  system: 'log-system',
};

export function ActivityTerminal({ lines }: ActivityTerminalProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <div className="activity-terminal">
      <div className="activity-terminal-scroll" ref={scrollRef}>
        {lines.length === 0 ? (
          <div className="activity-terminal-line">
            <span className="activity-prompt">$</span>
            <span className="activity-muted"> Waiting for stream...</span>
          </div>
        ) : (
          lines.map((line) => (
            <div key={line.id} className="activity-terminal-line">
              <span className="activity-prompt">&gt;</span>
              <span className="activity-time">{line.time}</span>
              <span className={TYPE_CLASS[line.type]}>{line.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
