"""
BlurBerry Screen Capture
Captures a monitor or a specific application window as a BGR numpy frame.
Same read() interface as cv2.VideoCapture so it's a drop-in for the pipeline.

Monitors: mss on every platform.
Windows:  the window's own pixels, even when other windows cover it: PrintWindow on
          Windows, Quartz (CGWindowList) on macOS. If that fails, the screen area the
          window occupies.

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
    import ctypes
    from ctypes import wintypes
    import win32gui

    class _BitmapInfoHeader(ctypes.Structure):
        _fields_ = [('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
                    ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD),
                    ('biCompression', wintypes.DWORD), ('biSizeImage', wintypes.DWORD),
                    ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
                    ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)]

    _user32, _gdi32, _dwmapi = ctypes.WinDLL('user32'), ctypes.WinDLL('gdi32'), ctypes.WinDLL('dwmapi')
    for _fn, _args, _res in (
        (_user32.IsIconic, [wintypes.HWND], wintypes.BOOL),
        (_user32.GetWindowRect, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        (_user32.GetWindowDC, [wintypes.HWND], wintypes.HDC),
        (_user32.ReleaseDC, [wintypes.HWND, wintypes.HDC], ctypes.c_int),
        (_user32.PrintWindow, [wintypes.HWND, wintypes.HDC, wintypes.UINT], wintypes.BOOL),
        (_gdi32.CreateCompatibleDC, [wintypes.HDC], wintypes.HDC),
        (_gdi32.CreateCompatibleBitmap, [wintypes.HDC, ctypes.c_int, ctypes.c_int], wintypes.HBITMAP),
        (_gdi32.SelectObject, [wintypes.HDC, wintypes.HGDIOBJ], wintypes.HGDIOBJ),
        (_gdi32.GetDIBits, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                            ctypes.POINTER(_BitmapInfoHeader), wintypes.UINT], ctypes.c_int),
        (_gdi32.DeleteObject, [wintypes.HGDIOBJ], wintypes.BOOL),
        (_gdi32.DeleteDC, [wintypes.HDC], wintypes.BOOL),
        (_dwmapi.DwmGetWindowAttribute, [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD],
         ctypes.c_long),
    ):
        _fn.argtypes, _fn.restype = _args, _res
    PW_RENDERFULLCONTENT = 2            # renders through DWM, so Chrome/Electron aren't black
    DWMWA_EXTENDED_FRAME_BOUNDS = 9     # the visible frame, without the invisible resize border
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

def _print_window(hwnd: int):
    """A Windows window's own pixels (BGR), even where other windows cover it, or None
    when Windows can't render it that way (minimised, some protected or GPU content)."""
    if _user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    window_dc = _user32.GetWindowDC(hwnd)
    memory_dc = _gdi32.CreateCompatibleDC(window_dc)
    bitmap = _gdi32.CreateCompatibleBitmap(window_dc, w, h)
    previous = _gdi32.SelectObject(memory_dc, bitmap)
    pixels = np.empty((h, w, 4), np.uint8)
    try:
        if not _user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT):
            return None
        header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        if not _gdi32.GetDIBits(memory_dc, bitmap, 0, h, pixels.ctypes.data, ctypes.byref(header), 0):
            return None
    finally:
        _gdi32.SelectObject(memory_dc, previous)
        _gdi32.DeleteObject(bitmap)
        _gdi32.DeleteDC(memory_dc)
        _user32.ReleaseDC(hwnd, window_dc)
    visible = wintypes.RECT()
    if _dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(visible),
                                     ctypes.sizeof(visible)) == 0:
        x1, y1 = max(0, visible.left - rect.left), max(0, visible.top - rect.top)
        x2, y2 = min(w, visible.right - rect.left), min(h, visible.bottom - rect.top)
        if x2 > x1 and y2 > y1:
            pixels = pixels[y1:y2, x1:x2]
    if not pixels[..., :3].any():
        return None
    return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)


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
        """Set which monitor or window to capture."""
        self._source = source

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
        if self.sct is None:
            self.sct = mss.mss()    # also makes the process DPI-aware, so sizes are in real pixels
        hwnd = self._source['hwnd']
        frame = _print_window(hwnd)
        if frame is not None:
            return True, _fit(frame)
        x, y, x2, y2 = win32gui.GetWindowRect(hwnd)
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
