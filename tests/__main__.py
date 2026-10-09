"""
Run every test:  python -m tests          (pytest tests works too)
                 python -m tests video    only tests/test_video.py
"""
import importlib
import os
import pkgutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wanted = sys.argv[1:]
failed, passed = [], 0
for info in sorted(pkgutil.iter_modules([os.path.dirname(os.path.abspath(__file__))]), key=lambda m: m.name):
    if not info.name.startswith('test_') or (wanted and info.name[5:] not in wanted):
        continue
    module = importlib.import_module(f'tests.{info.name}')
    for name, fn in list(vars(module).items()):
        if not (name.startswith('test_') and callable(fn) and fn.__module__ == module.__name__):
            continue
        label = f'{info.name}.{name}'
        t0 = time.perf_counter()
        try:
            fn()
            passed += 1
            print(f'PASS {label} ({time.perf_counter() - t0:.1f}s)', flush=True)
        except Exception:
            failed.append(label)
            print(f'FAIL {label}\n{traceback.format_exc()}', flush=True)

print(f'\n{passed} passed, {len(failed)} failed' + (': ' + ', '.join(failed) if failed else ''))
sys.exit(1 if failed else 0)
