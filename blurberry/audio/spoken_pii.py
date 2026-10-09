"""
Spoken PII detector for a live ASR word stream.

ASR output is lowercase words ("my number is nine eight seven ..."), so the words are
first normalised into units: spoken numbers become digits, spelled letters become
capitals, "at the rate" becomes "@". Three kinds of rule then flag words:
  runs      dictated digits, long spoken numbers, spelled IDs (no context needed)
  patterns  regexes over the normalised text (emails, UPI IDs, PAN, IFSC, plates)
  triggers  a phrase such as "my phone number is" arms redaction of what follows, so
            the payload is caught on its first word rather than once enough of it
            has been heard to look like PII

detect() is stateless: the caller re-runs it over the recent words on every update.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Word:
    text: str
    start: float
    end: float
    timed: bool = True      # False when start/end are estimates


UNITS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
         'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
TEENS = {'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
         'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
SCALES = {'thousand': 1000, 'lakh': 100_000, 'lac': 100_000, 'million': 1_000_000, 'crore': 10_000_000}
REPEATS = {'double': 2, 'triple': 3}
# ASR spellings of digits; only read as digits between two other numbers
HOMOPHONES = {'to': 'two', 'too': 'two', 'for': 'four', 'won': 'one', 'ate': 'eight', 'tree': 'three'}
ZEROS = {'oh', 'o'}
LETTER_NAMES = {'bee': 'B', 'cee': 'C', 'dee': 'D', 'ef': 'F', 'eff': 'F', 'gee': 'G', 'aitch': 'H',
                'jay': 'J', 'kay': 'K', 'el': 'L', 'em': 'M', 'en': 'N', 'pee': 'P', 'cue': 'Q',
                'ar': 'R', 'ess': 'S', 'tee': 'T', 'vee': 'V', 'ex': 'X', 'wye': 'Y', 'zed': 'Z', 'zee': 'Z'}
FILLERS = {'uh', 'um', 'umm', 'er', 'ah', 'eh', 'hmm', 'mm'}
RUN_GLUE = FILLERS | {'dash', 'hyphen', 'plus'}
RATE_WORDS = {'rate', 'rates', 'rage'}
CONNECTORS = FILLERS | {'is', "it's", 'its', 'it', 'was', 'are', 'as', 'the', 'my', 'number', 'no',
                        'code', 'id', 'colon', 'like', 'basically', 'actually', 'again'}
STOP = {'and', 'but', 'so', 'okay', 'ok', 'then', 'also', 'please', 'thanks', 'thank', 'right',
        'yeah', 'yes', 'anyway', 'because'}
NAME_STOP = STOP | FILLERS | {'i', "i'm", 'the', 'a', 'an', 'not', 'is', 'was', 'from', 'in', 'at',
                              'of', 'to', 'for', 'with', 'here', 'there', 'speaking', 'calling',
                              'just', 'really', 'very', 'actually', 'my', 'and'}
EMAIL_STOP = STOP | {'is', 'my', 'me', 'email', 'mail', 'id', 'address', 'its', "it's", 'the',
                     'to', 'on', 'send', 'write', 'at'}
WEB_TLDS = {'com', 'org', 'net', 'io', 'ai', 'edu', 'gov'}
MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
          'september', 'october', 'november', 'december'}
ORDINALS = {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
            'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth',
            'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'thirtieth'}
DATE_WORDS = MONTHS | ORDINALS | {'of', 'the', 'st', 'nd', 'rd', 'th'}

MIN_DICTATED = 3        # "nine eight seven": digit-by-digit dictation is almost always PII
MIN_GROUPS = 4          # "ninety eight forty five sixty seven twenty one"
MIN_SPELLED = 4         # "j o h n"
TRIGGER_WINDOW = 4.0    # seconds from a trigger to the start of its payload
MAX_GAP = {'alnum': 2.0, 'secret': 1.2, 'email': 2.0, 'name': 0.8, 'address': 1.5, 'date': 1.5}

_AT = r"(?: at | ?@ ?)"
_DOT = r"(?: dot |\.)"
_PROVIDERS = r"(?:gmail|g mail|yahoo|hotmail|outlook|icloud|rediffmail|rediff|protonmail|proton|ymail)"
_TLD = r"(?:com|in|org|net|io|ai|edu|gov|co)"

TRIGGERS = [(label, mode, re.compile(rx)) for label, mode, rx in [
    ('phone', 'alnum', r"\b(?:phone|mobile|cell|whatsapp|contact|telephone|landline)(?: number| no)?\b"
                       r"|\b(?:call|reach|text|ping|contact) me (?:on|at)\b|\bmy number\b"),
    ('aadhaar', 'alnum', r"\b(?:aadhaar|aadhar|adhaar|adhar)(?: card)?(?: number| no)?\b"),
    ('card', 'alnum', r"\b(?:card|credit card|debit card|atm card)(?: number| no)\b"
                      r"|\b(?:cvv|c v v|cvc|expiry|expiration)(?: date| number| code)?\b"),
    ('account', 'alnum', r"\b(?:account|acc|a c|bank account)(?: number| no)\b"
                         r"|\b(?:ifsc|i f s c|swift|routing)(?: code| number)?\b|\bupi(?: id)?\b"),
    ('id', 'alnum', r"\b(?:pan|pen|passport|license|licence|voter|vehicle|registration|gst|gstin|ssn)"
                    r"(?: card)?(?: number| no| id)?\b"),
    ('otp', 'alnum', r"\b(?:otp|o t p|one time password|pin|pin code|pincode|zip code|passcode|pass code"
                     r"|verification code|security code|code)\b"),
    ('number', 'alnum', r"\bnumbers?\b"),
    ('password', 'secret', r"\b(?:password|pass word|passwd|secret word|login)\b"),
    ('email', 'email', r"\b(?:e ?mail|mail)(?: id| address)?\b"),
    ('name', 'name', r"\b(?:my name is|my name's|name is|i am called|i'm called)\b"),
    ('address', 'address', r"\b(?:my address|address is|i live (?:at|in|on)|i stay (?:at|in)"
                           r"|located at|my house is|house number|flat number)\b"),
    ('dob', 'date', r"\b(?:date of birth|born on|birthday is|d o b|dob)\b"),
]]

PATTERNS = [(label, re.compile(rx, flags)) for label, rx, flags in [
    ('email', r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", 0),
    ('email', rf"[\w+-]+(?:{_DOT}[\w+-]+)*{_AT}{_PROVIDERS}\b(?:{_DOT}\w+)*", re.I),
    ('email', rf"[\w+-]+(?:{_DOT}[\w+-]+)*{_AT}[\w-]+(?:{_DOT}[\w-]+)*{_DOT}{_TLD}\b", re.I),
    ('upi', rf"[\w.+-]+{_AT}(?:ok\w+|ybl|ibl|axl|paytm|upi|apl)\b", re.I),
    ('pan', r"\b[A-Z]{5}\d{4}[A-Z]\b", 0),
    ('ifsc', r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0),
    ('plate', r"\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}\b", 0),
]]


@dataclass
class _Unit:
    kind: str           # num | alpha | word | filler
    text: str
    i0: int             # words[i0:i1] make up this unit
    i1: int
    dictated: int = 0   # digits spoken one by one


def _clean(text: str) -> str:
    return re.sub(r"^[^\w@]+|[^\w@]+$", "", text.lower().strip())


def _is_numeric(t: str) -> bool:
    return t in UNITS or t in TEENS or t in TENS or t in REPEATS or t.isdigit()


def _canonical(words: List[Word]) -> List[str]:
    toks = [_clean(w.text) for w in words]

    def numeric_run(i, step):
        n = 0
        while 0 <= i < len(toks) and _is_numeric(toks[i]):
            n, i = n + 1, i + step
        return n

    for i, t in enumerate(toks):
        if t in HOMOPHONES:
            left, right = numeric_run(i - 1, -1), numeric_run(i + 1, 1)
            if left >= 2 or (left and right and left + right >= 3):
                toks[i] = HOMOPHONES[t]
        elif t in ZEROS and (numeric_run(i - 1, -1) or numeric_run(i + 1, 1)):
            toks[i] = 'zero'
    return toks


def _tens(toks, i) -> Tuple[Optional[int], int]:
    t = toks[i] if i < len(toks) else ''
    if t in TENS:
        if i + 1 < len(toks) and UNITS.get(toks[i + 1], 0) > 0:
            return TENS[t] + UNITS[toks[i + 1]], i + 2
        return TENS[t], i + 1
    if t in TEENS:
        return TEENS[t], i + 1
    if UNITS.get(t, 0) > 0:
        return UNITS[t], i + 1
    return None, i


def _below_thousand(toks, i) -> Tuple[Optional[int], int, bool]:
    """(value, next index, compound). A lone unit digit is not compound."""
    t = toks[i] if i < len(toks) else ''
    if t in UNITS or t in TEENS:
        v, j = UNITS.get(t, TEENS.get(t)), i + 1
        if t != 'zero' and j < len(toks) and toks[j] == 'hundred':
            v, j = v * 100, j + 1
            if j + 1 < len(toks) and toks[j] == 'and' and _tens(toks, j + 1)[0] is not None:
                j += 1
            r, j2 = _tens(toks, j)
            return (v + r, j2, True) if r is not None else (v, j, True)
        return v, j, t in TEENS
    if t in TENS:
        v, j = _tens(toks, i)
        return v, j, True
    return None, i, False


def _parse_number(toks, i) -> Optional[Tuple[str, int, int]]:
    """(digits, next index, dictated digit count) for a spoken number at toks[i]."""
    t = toks[i]
    if t.isdigit():
        return t, i + 1, len(t) if len(t) >= 5 else 0
    if t in REPEATS and i + 1 < len(toks) and toks[i + 1] in UNITS:
        k = REPEATS[t]
        return str(UNITS[toks[i + 1]]) * k, i + 2, k
    v, j, compound = _below_thousand(toks, i)
    if v is None:
        return None
    if j < len(toks) and toks[j] in SCALES:
        v, j, compound = v * SCALES[toks[j]], j + 1, True
        k = j + 1 if j + 1 < len(toks) and toks[j] == 'and' else j
        r, j2, _ = _below_thousand(toks, k)
        if r is not None:
            v, j = v + r, j2
    return str(v), j, 0 if compound else 1


def _units(toks: List[str]) -> List[_Unit]:
    units, i = [], 0
    while i < len(toks):
        t = toks[i]
        num = _parse_number(toks, i) if t else None
        if num:
            digits, j, dictated = num
            units.append(_Unit('num', digits, i, j, dictated))
            i = j
            continue
        if t == 'at' and i + 2 < len(toks) and toks[i + 1] == 'the' and toks[i + 2] in RATE_WORDS:
            units.append(_Unit('word', '@', i, i + 3))
            i += 3
            continue
        if t in FILLERS:
            units.append(_Unit('filler', t, i, i + 1))
        elif len(t) == 1 and t.isalpha():
            units.append(_Unit('letter', t.upper(), i, i + 1))
        elif t in LETTER_NAMES:
            units.append(_Unit('letter', LETTER_NAMES[t], i, i + 1))
        elif re.fullmatch(r"[a-z0-9]+", t) and re.search(r"\d", t) and re.search(r"[a-z]", t):
            units.append(_Unit('alpha', t.upper(), i, i + 1))
        else:
            units.append(_Unit('word', t, i, i + 1))
        i += 1

    # Single letters and letter names are only letters when spelling something:
    # next to another letter or a number. "a", "i" and names like "jay" need that
    # neighbour to be a clear letter or a number, not another ambiguous word.
    def solid(k):
        u = units[k] if 0 <= k < len(units) else None
        if u is None:
            return False
        if u.kind in ('num', 'alpha'):
            return True
        return u.kind == 'letter' and toks[u.i0] not in ('a', 'i') and toks[u.i0] not in LETTER_NAMES

    resolved = []
    for k, u in enumerate(units):
        if u.kind == 'letter':
            strong = toks[u.i0] not in ('a', 'i') and toks[u.i0] not in LETTER_NAMES
            near = solid(k - 1) or solid(k + 1)
            u = _Unit('alpha', u.text, u.i0, u.i1) if strong or near else _Unit('word', toks[u.i0], u.i0, u.i1)
        resolved.append(u)
    return resolved


def _compose(units: List[_Unit]) -> Tuple[str, List[int]]:
    """Units as one string for the regexes, with the unit index of every character.
    Neighbouring digits and letters are glued, so "a b c d e 1 2 3 4 f" reads ABCDE1234F."""
    text, owner = '', []
    for k, u in enumerate(units):
        glued = k and units[k - 1].kind in ('num', 'alpha') and u.kind in ('num', 'alpha')
        if text and not glued:
            text += ' '
            owner.append(-1)
        text += u.text
        owner.extend([k] * len(u.text))
    return text, owner


def _runs(units: List[_Unit]) -> List[List[int]]:
    runs, cur = [], []
    for k, u in enumerate(units):
        if u.kind in ('num', 'alpha'):
            cur.append(k)
        elif cur and (u.kind == 'filler' or u.text in RUN_GLUE):
            continue
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _classify_run(units: List[_Unit], run: List[int]) -> Optional[str]:
    nums = [units[k] for k in run if units[k].kind == 'num']
    alphas = [units[k] for k in run if units[k].kind == 'alpha']
    dictated = sum(u.dictated for u in nums)
    if any(re.search(r"\d", u.text) for u in alphas):
        return 'id'
    if dictated >= MIN_DICTATED or len(nums) >= MIN_GROUPS:
        return 'id' if alphas else 'number'
    if len(alphas) >= MIN_SPELLED:
        return 'spelled'
    if len(alphas) >= 2 and len(nums) >= 2:
        return 'id'
    return None


def _payload(units: List[_Unit], words: List[Word], k: int, mode: str, t_trigger: float) -> List[int]:
    """Units armed by a trigger that ends just before units[k]."""
    taken, plain, counted = [], 0, 0
    last = t_trigger
    lead = CONNECTORS | EMAIL_STOP if mode == 'email' else CONNECTORS
    for m in range(k, len(units)):
        u = units[m]
        start = words[u.i0].start
        if start - last > (MAX_GAP[mode] if taken else TRIGGER_WINDOW):
            break
        if not taken and u.text in lead:
            last = words[u.i1 - 1].end
            continue

        if mode == 'alnum':
            if u.kind in ('num', 'alpha'):
                taken.append(m)
            elif u.kind == 'filler' or u.text in RUN_GLUE:
                pass
            elif taken:
                break
            else:
                plain += 1
                if plain > 1:
                    break
        elif mode == 'date':
            if u.kind in ('num', 'alpha') or u.text in DATE_WORDS:
                taken.append(m)
                if len(taken) >= 6:
                    break
            elif taken or u.kind != 'filler':
                break
        else:
            if taken and u.text in STOP:
                break
            if mode == 'name' and u.kind == 'word' and u.text in NAME_STOP:
                break
            if u.kind != 'filler':
                taken.append(m)
                counted += u.kind == 'word'
            if mode == 'secret' and counted >= 3:
                break
            if mode == 'name' and counted >= 2:
                break
            if mode == 'address' and len(taken) >= 12:
                break
            if mode == 'email':
                prev = units[m - 1].text if m else ''
                if (prev == 'dot' and u.text in WEB_TLDS | {'in'}) or len(taken) >= 10:
                    break
        last = words[u.i1 - 1].end
    return taken


def _web_tails(units: List[_Unit]) -> List[List[int]]:
    """'... dot com': the address words just before a TLD, however mangled by the ASR."""
    out = []
    for k in range(1, len(units)):
        if units[k].text in WEB_TLDS and units[k - 1].text == 'dot':
            span = [k - 1, k]
            j = k - 2
            while j >= 0 and len(span) < 8 and units[j].text not in EMAIL_STOP:
                span.append(j)
                j -= 1
            out.append(span)
    return out


def detect(words: List[Word]) -> Dict[int, str]:
    """Indices of words that should be redacted, mapped to a PII label."""
    if not words:
        return {}
    units = _units(_canonical(words))
    text, owner = _compose(units)
    flags: Dict[int, str] = {}

    def flag(ks, label):
        for k in ks:
            if units[k].kind != 'filler':
                flags.setdefault(k, label)

    for label, mode, rx in TRIGGERS:
        for m in rx.finditer(text):
            k = owner[m.end() - 1]
            flag(_payload(units, words, k + 1, mode, words[units[k].i1 - 1].end), label)
    for label, rx in PATTERNS:
        for m in rx.finditer(text):
            flag({owner[c] for c in range(m.start(), m.end()) if owner[c] >= 0}, label)
    for ks in _web_tails(units):
        flag(ks, 'email')
    for run in _runs(units):
        label = _classify_run(units, run)
        if label:
            flag(run, label)

    return {i: label for k, label in flags.items() for i in range(units[k].i0, units[k].i1)}


def words_from_text(text: str, step: float = 0.3) -> List[Word]:
    """Evenly timed words from a sentence, for tests and the CLI."""
    return [Word(w, i * step, i * step + step * 0.9) for i, w in enumerate(text.split())]


def redact_text(text: str) -> str:
    words = words_from_text(text)
    flags = detect(words)
    return ' '.join(f'[{flags[i].upper()}]' if i in flags else w.text for i, w in enumerate(words))
