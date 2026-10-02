#!/usr/bin/env python3
"""
Build a standalone Windows .exe for the current SoloTone version, using
PyInstaller. Run from anywhere:

    pip install -r build/requirements-build.txt
    python build/build_exe.py

Output lands in build/dist/SoloTone-v<VERSION_FULL>.exe

Deliberately NOT bundled into the .exe itself: the website/ folder, sample
nam/ir content, or anything under build/ — this packages the app only.
nam/ and irs/ folders are created next to the .exe on first launch, same
as when run from source (see _ensure_dirs() / _APP_DIR in solotone.py).

SoloTone_User_Guide.docx IS copied into dist/ as a plain sibling file next
to the .exe (not embedded in the binary) — so it opens directly in Word
without launching the app.
"""
import pathlib
import shutil
import sys

import PyInstaller.__main__

ROOT  = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'

sys.path.insert(0, str(ROOT))
from version import VERSION_FULL  # noqa: E402

APP_NAME  = f'SoloTone-v{VERSION_FULL}'
ENTRY     = ROOT / 'solotone.py'
ICON      = BUILD / 'solotone.ico'
DIST_DIR  = BUILD / 'dist'
WORK_DIR  = BUILD / 'work'
SPEC_DIR  = BUILD / 'spec'
USER_GUIDE = ROOT / 'SoloTone_User_Guide.docx'


def main():
    args = [
        str(ENTRY),
        '--name', APP_NAME,
        '--onefile',
        '--windowed',                 # no console window for this GUI app
        '--icon', str(ICON),
        '--distpath', str(DIST_DIR),
        '--workpath', str(WORK_DIR),
        '--specpath', str(SPEC_DIR),
        '--noconfirm',
        # No --add-data for website/, nam/, or irs/: those stay out of the
        # bundle on purpose. nam/ and irs/ are created next to the .exe at
        # first launch instead.
    ]
    PyInstaller.__main__.run(args)

    exe = DIST_DIR / f'{APP_NAME}.exe'
    if exe.exists():
        print(f'\nBuilt {exe}  ({exe.stat().st_size / 1_048_576:.1f} MB)')
    else:
        print('\nBuild finished but the expected .exe was not found — check the log above.')

    if USER_GUIDE.exists():
        shutil.copy2(USER_GUIDE, DIST_DIR / USER_GUIDE.name)
        print(f'Copied {USER_GUIDE.name} into {DIST_DIR}')
    else:
        print(f'WARNING: {USER_GUIDE} not found — user guide not copied into dist/')

    # PyInstaller's intermediate work/ and spec/ folders aren't useful once
    # the exe is built; leave dist/ as the only lasting output.
    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == '__main__':
    main()
