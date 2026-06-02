"""Smoke test: every in-scope module must import under the stub."""
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ORTHANC_DIR = os.path.abspath(os.path.join(_THIS, '..', '..'))


def _scoped_import(target_dir, dotted_names):
    """Pin target_dir at sys.path[0], import dotted_names, then restore sys.path.

    Each name in dotted_names is also evicted from sys.modules before and after
    so back-to-back smoke tests don't cross-contaminate cached modules.
    """
    saved_path = list(sys.path)
    if target_dir in sys.path:
        sys.path.remove(target_dir)
    sys.path.insert(0, target_dir)
    # evict any previously-cached top-level names we are about to import
    for name in dotted_names:
        top = name.split('.', 1)[0]
        for k in list(sys.modules):
            if k == top or k.startswith(top + '.'):
                del sys.modules[k]
    try:
        for name in dotted_names:
            __import__(name)
    finally:
        sys.path[:] = saved_path


def test_viewer_modules_import():
    _scoped_import(
        os.path.join(_ORTHANC_DIR, 'viewer'),
        ['feedback_db', 'feedback_routes', 'router'],
    )


def test_router_modules_import():
    _scoped_import(
        os.path.join(_ORTHANC_DIR, 'router'),
        ['server', 'wado_utils', 'ups.routes', 'ups.processor',
         'ups.storage', 'ups.subscription_storage', 'ups.workitem'],
    )
