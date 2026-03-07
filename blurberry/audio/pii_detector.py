import re
import spacy
from better_profanity import profanity
from typing import List, Tuple
import os


class PIIDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load custom profanity wordlist
        custom_slurs_path = os.path.join(os.path.dirname(__file__), "..", "data", "wordlists", "custom_slurs.txt")
        try:
            with open(custom_slurs_path, 'r') as f:
                custom_words = [line.strip().lower() for line in f if line.strip()]
                profanity.add_censor_words(custom_words)
        except FileNotFoundError:
            pass
        
        # Regex patterns for PII detection
        self.patterns = {
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            'indian_phone': r'\b(?:\+91|0)?[6-9]\d{9}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'bank_account': r'\b\d{10,18}\b',
            'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            'upi_id': r'\b[A-Za-z0-9._-]+@[a-z]{2,}\b',
            'spoken_password': r'\b(?:my password|your password|the password|password is|pass is|secret is|key is)\s+[^.!?]*\b',
            'address_keywords': r'\b(?:my address|your address|the address|address is|i live|live at|from|located at|street|road|apartment|flat|house)\b'
        }
        
        # Sensitive context words for NER
        self.context_words = [
            'my name is', 'i am', 'my name', 'name is', 'i live', 'live at', 
            'address', 'from', 'my address', 'your address', 'the address',
            'my company', 'work at', 'office at', 'located at', 'based in'
        ]
    
    def detect_pii(self, text: str) -> List[Tuple[str, str]]:
        findings = []
        text_lower = text.lower()
        
        # Regex-based detection
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append((pii_type, match.group()))
        
        # Profanity detection
        if profanity.contains_profanity(text):
            profane_words = []
            words = text_lower.split()
            for word in words:
                if profanity.contains_profanity(word):
                    profane_words.append(word)
            if profane_words:
                findings.append(('profanity', ', '.join(profane_words)))
        
        # spaCy NER with context filtering
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG', 'FAC']:
                # Check if entity is preceded by sensitive context
                entity_start = ent.start_char
                context_window = text_lower[max(0, entity_start-50):entity_start]
                
                has_sensitive_context = any(ctx_word in context_window for ctx_word in self.context_words)
                
                if has_sensitive_context:
                    findings.append((f'ner_{ent.label_.lower()}', ent.text))
        
        return findings
    
    def contains_pii(self, text: str) -> bool:
        return len(self.detect_pii(text)) > 0