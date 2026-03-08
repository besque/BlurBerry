import { useState } from 'react';
import { FaceCaptureModal } from './FaceCaptureModal';
import './EnrollFacePage.css';

interface EnrollFacePageProps {
  onBack: () => void;
  onSkipAndFinish: () => void;
  onCaptureComplete: () => void;
}

export function EnrollFacePage({ onBack, onSkipAndFinish, onCaptureComplete }: EnrollFacePageProps) {
  const [modalOpen, setModalOpen] = useState(false);

  const handleStartRegistration = () => {
    setModalOpen(true);
  };

  const handleCaptureComplete = () => {
    setModalOpen(false);
    onCaptureComplete();
  };

  return (
    <>
      <div className="enroll-face-page">
        <h1 className="enroll-face-title">Register Your Face</h1>
        <p className="enroll-face-subtitle">
          Optional: Register your face to enable whitelist mode. You won&apos;t be blurred, but everyone else will be.
        </p>

        <div className="enroll-face-icon-wrap">
          <div className="enroll-face-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
        </div>

        <button type="button" className="enroll-face-cta" onClick={handleStartRegistration}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M23 7l-7 5 7 5V7z" />
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
          </svg>
          Start Face Registration
        </button>

        <p className="enroll-face-privacy">
          Your face data is stored locally on your device and never uploaded to the cloud.
        </p>

        <div className="enroll-face-dots">
          <span className="enroll-face-dot" />
          <span className="enroll-face-dot" />
          <span className="enroll-face-dot" />
          <span className="enroll-face-dot active" />
        </div>

        <div className="enroll-face-nav">
          <button type="button" className="enroll-face-back" onClick={onBack}>
            Back
          </button>
          <button type="button" className="enroll-face-skip" onClick={onSkipAndFinish}>
            Skip &amp; Finish
          </button>
        </div>
      </div>

      {modalOpen && (
        <FaceCaptureModal
          onClose={() => setModalOpen(false)}
          onCaptureComplete={handleCaptureComplete}
          onSkipAndFinish={onSkipAndFinish}
        />
      )}
    </>
  );
}
