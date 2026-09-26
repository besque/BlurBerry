"""
BlurBerry Screen Capture
Captures a monitor or a specific application window as a BGR numpy frame.
Same read() interface as cv2.VideoCapture so it's a drop-in for the pipeline.

Monitors: mss on every platform.
Windows:  Win32 on Windows; Quartz (CGWindowList) on macOS, which captures the
          window's own pixels even when other windows cover it.

macOS needs Screen Recording permission for the app that launches the server
(System Settings → Privacy & Security → Screen Recording); without it captures
come back black or show only the wallpaper.

Ported from CenServe (censerve/video/screen_capture.py).
"""

import sys
import numpy as np
import cv2
import mss

OUT_W, OUT_H = 1280, 720

if sys.platform == "win32":
    import win32gui
elif sys.platform == "darwin":
    import Quartz


# ── Source enumeration ────────────────────────────────────────────────────────

def _list_windows_win32() -> list:
    def _enum_cb(hwnd, out):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title or len(title) < 2:
            return
        rect = win32gui.GetWindowRect(hwnd)
        if rect[2] - rect[0] < 200 or rect[3] - rect[1] < 150:
            return
        out.append({'id': f'window_{hwnd}', 'label': title, 'type': 'window', 'hwnd': hwnd})

    wins = []
    win32gui.EnumWindows(_enum_cb, wins)
    return wins


def _list_windows_macos() -> list:
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    infos = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
    wins = []
    for info in infos:
        if info.get('kCGWindowLayer', 1) != 0:       # skip menu bar, dock, overlays
            continue
        b = info.get('kCGWindowBounds', {})
        if b.get('Width', 0) < 200 or b.get('Height', 0) < 150:
            continue
        owner = info.get('kCGWindowOwnerName', '') or ''
        title = info.get('kCGWindowName', '') or ''   # empty without Screen Recording permission
        if owner in ('Window Server', 'Dock'):
            continue
        wid = int(info['kCGWindowNumber'])
        label = f'{owner} — {title}' if title else owner
        wins.append({'id': f'window_{wid}', 'label': label, 'type': 'window', 'wid': wid})
    return wins


def list_sources() -> list:
    """
    Returns all capturable sources — monitors and visible application windows.
    Each source is a dict with:
      id      str   unique identifier
      label   str   human-readable name shown in the UI
      type    str   'monitor' or 'window'
    Monitors also carry 'index' (mss, 1-based); windows carry 'hwnd' (Windows)
    or 'wid' (macOS window number).
    """
    sources = []
    with mss.mss() as sct:
        for i, mon in enumerate(sct.monitors[1:], 1):
            sources.append({
                'id':    f'monitor_{i}',
                'label': f'Monitor {i}  ({mon["width"]}×{mon["height"]})',
                'type':  'monitor',
                'index': i,
            })

    if sys.platform == "win32":
        sources.extend(_list_windows_win32()[:20])
    elif sys.platform == "darwin":
        sources.extend(_list_windows_macos()[:20])
    return sources


# ── Capture class ─────────────────────────────────────────────────────────────

def _fit(frame: np.ndarray) -> np.ndarray:
    """Letterbox into OUT_W×OUT_H without stretching."""
    h, w = frame.shape[:2]
    s = min(OUT_W / w, OUT_H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    out = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    x0, y0 = (OUT_W - nw) // 2, (OUT_H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


class ScreenCapture:
    """
    Captures frames from a selected monitor or application window.

    Usage:
        sc = ScreenCapture()
        sc.set_source({'type': 'monitor', 'index': 1})
        ok, frame = sc.read()   # frame is BGR numpy array, 1280×720
    """

    def __init__(self):
        self.sct     = None     # created lazily on the capturing thread (mss is thread-bound)
        self._source = None

    def set_source(self, source: dict):
        """
        Set which monitor or window to capture.
        On Windows, also calls SetWindowDisplayAffinity so GPU-accelerated windows
        (Chrome, Electron) don't capture as black frames.
        """
        self._source = source
        if sys.platform == "win32" and source.get('type') == 'window' and source.get('hwnd'):
            try:
                win32gui.SetWindowDisplayAffinity(source['hwnd'], 0)
            except Exception as e:
                print(f'[ScreenCapture] SetWindowDisplayAffinity failed: {e}')
                print('[ScreenCapture] Chrome/Electron tabs may capture as black')

    def read(self):
        """
        Capture one frame from the selected source.
        Returns (True, frame_bgr) on success, (False, None) on failure.
        Output is letterboxed to 1280×720 for consistent pipeline input.
        """
        if self._source is None:
            return False, None
        try:
            if self._source['type'] == 'monitor':
                return self._read_monitor()
            if self._source['type'] == 'window':
                if sys.platform == "darwin":
                    return self._read_window_macos()
                if sys.platform == "win32":
                    return self._read_window_win32()
            return False, None
        except Exception as e:
            print(f'[ScreenCapture] read() error: {e}')
            return False, None

    def _grab(self, region):
        if self.sct is None:
            self.sct = mss.mss()
        return cv2.cvtColor(np.array(self.sct.grab(region)), cv2.COLOR_BGRA2BGR)

    def _read_monitor(self):
        if self.sct is None:
            self.sct = mss.mss()
        return True, _fit(self._grab(self.sct.monitors[self._source['index']]))

    def _read_window_win32(self):
        x, y, x2, y2 = win32gui.GetWindowRect(self._source['hwnd'])
        if x2 <= x or y2 <= y:
            return False, None
        return True, _fit(self._grab({'left': x, 'top': y, 'width': x2 - x, 'height': y2 - y}))

    def _read_window_macos(self):
        img = None
        try:
            # Deprecated by Apple in macOS 15 but still the only synchronous per-window capture
            img = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionIncludingWindow,
                self._source['wid'],
                Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution,
            )
        except Exception:
            pass
        if img is None:
            return self._read_window_bounds_macos()
        w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
        if w == 0 or h == 0:
            return False, None
        stride = Quartz.CGImageGetBytesPerRow(img)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        bgra = np.frombuffer(data, np.uint8).reshape(h, stride // 4, 4)[:, :w]
        return True, _fit(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))

    def _read_window_bounds_macos(self):
        """Fallback: grab the screen area the window occupies (it must be visible)."""
        infos = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionIncludingWindow, self._source['wid']) or []
        if not infos:
            return False, None
        b = infos[0].get('kCGWindowBounds', {})
        region = {'left': int(b.get('X', 0)), 'top': int(b.get('Y', 0)),
                  'width': int(b.get('Width', 0)), 'height': int(b.get('Height', 0))}
        if region['width'] <= 0 or region['height'] <= 0:
            return False, None
        return True, _fit(self._grab(region))

    def release(self):
        try:
            if self.sct is not None:
                self.sct.close()
        except Exception:
            pass
