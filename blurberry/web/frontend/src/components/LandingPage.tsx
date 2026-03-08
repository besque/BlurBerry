import './LandingPage.css';

interface LandingPageProps {
  onStartEnrollment: () => void;
  onStartStream: () => void;
}

export function LandingPage({ onStartEnrollment, onStartStream }: LandingPageProps) {
  return (
    <div className="landing">
      <div className="landing-card">
        <div className="logo">
          <div className="logo-icon">🫐</div>
          BlurBerry <span>AI</span>
        </div>
        <p className="landing-tagline">Privacy filter for your stream</p>
        <div className="landing-actions">
          <button type="button" className="btn btn-primary btn-landing" onClick={onStartEnrollment}>
            Enroll my face
          </button>
          <button type="button" className="btn btn-primary btn-landing btn-secondary-cta" onClick={onStartStream}>
            Start stream
          </button>
        </div>
        <p className="landing-hint">
          Enroll your face so it won’t be blurred, or go straight to the stream dashboard and start your camera.
        </p>
      </div>
    </div>
  );
}
