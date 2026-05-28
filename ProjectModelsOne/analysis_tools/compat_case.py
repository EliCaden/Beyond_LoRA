# analysis_tools/compat_case.py
import os, sys, argparse

def _resolve_ci(path: str) -> str:
    """Resolve a potentially mis-cased relative/absolute path to the path on disk."""
    if not path:
        return path
    if os.path.exists(path):
        return path

    # Make absolute base
    if os.path.isabs(path):
        cur = os.sep
        parts = os.path.normpath(path).split(os.sep)[1:]  # drop leading ''
    else:
        cur = os.getcwd()
        parts = os.path.normpath(path).split(os.sep)

    for part in parts:
        try:
            entries = os.listdir(cur)
        except FileNotFoundError:
            return path  # give up; argparse will error as usual
        match = next((e for e in entries if e.lower() == part.lower()), None)
        if match is None:
            return path
        cur = os.path.join(cur, match)
    return cur

# Patch argparse so ALL @file includes (including nested ones) are case-folded.
_orig = argparse.ArgumentParser._read_args_from_files

def _ci_read_args_from_files(self, arg_strings):
    folded = []
    for a in arg_strings:
        if isinstance(a, str) and a.startswith("@") and len(a) > 1:
            folded.append("@" + _resolve_ci(a[1:]))
        else:
            folded.append(a)
    return _orig(self, folded)

argparse.ArgumentParser._read_args_from_files = _ci_read_args_from_files

# Also fold top-level sys.argv once (handles the very first @file on the CLI)
sys.argv = [sys.argv[0]] + [
    ("@" + _resolve_ci(a[1:]) if isinstance(a, str) and a.startswith("@") else a)
    for a in sys.argv[1:]
]
