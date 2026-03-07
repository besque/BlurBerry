import torch
import numpy as np
from typing import List, Tuple


class VAD:
    def __init__(self):
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        torch.hub.set_dir(torch.hub.get_dir() or torch.hub._get_torch_home())
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.validate_url, self.get_sampling_rate) = self.utils
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        if self.model is None:
            return False
        
        if len(audio_chunk) == 0:
            return False
        
        # Convert numpy array to tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk
        
        # Ensure the audio is in the right format
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        try:
            speech_prob = self.model(audio_tensor, 16000).item()
            return speech_prob > 0.4
        except Exception:
            return False
    
    def get_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        if self.model is None:
            return []
        
        if len(audio) == 0:
            return []
        
        # Convert numpy array to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        try:
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=16000)
            
            # Convert to (start_sample, end_sample) tuples
            segments = []
            for timestamp in speech_timestamps:
                start_sample = timestamp['start']
                end_sample = timestamp['end']
                segments.append((start_sample, end_sample))
            
            return segments
        except Exception:
            return []