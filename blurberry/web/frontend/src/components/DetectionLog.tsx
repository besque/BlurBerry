import type { LogEntry } from '../hooks/useBlurBerry';

interface DetectionLogProps {
  entries: LogEntry[];
  terminal?: boolean;
}

const LOG_CLASS: Record<string, string> = {
  face: 'log-face',
  plate: 'log-plate',
  text_pii: 'log-pii',
  nsfw: 'log-nsfw',
  card: 'log-card',
  audio_pii: 'log-pii',
};

export function DetectionLog({ entries, terminal }: DetectionLogProps) {
  const className = terminal ? 'log-box terminal-log' : 'log-box';
  return (
    <div className={className}>
      {entries.length === 0 ? (
        <div className="log-line">
          <span className="log-prompt">$</span>
          <span className="log-muted"> Waiting for stream...</span>
        </div>
      ) : (
        entries.map((e) => (
          <div key={e.id} className="log-line">
            <span className="log-prompt">&gt;</span>
            <span className="log-time">{e.time}</span>
            <span className={LOG_CLASS[e.type] || 'log-muted'}>{e.message}</span>
          </div>
        ))
      )}
    </div>
  );
}
