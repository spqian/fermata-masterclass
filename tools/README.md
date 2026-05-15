# Bundled Toolchain

Fermata Masterclass depends on several large third-party tools that are **not committed to the repository**. They are downloaded on demand into this directory.

## What lives here (after install)

| Folder | Purpose | Approx size | License |
|---|---|---|---|
| `python/` | Embedded CPython 3.11 | ~30 MB | PSF |
| `ffmpeg/` | Audio + video extraction (`ffmpeg.exe`, `ffprobe.exe`) | ~150 MB | LGPL-2.1+ |
| `audiveris/` | Audiveris 5.6.2 OMR — PDF → MusicXML | ~250 MB | AGPL-3.0 |
| `jre/` | Eclipse Temurin JDK 21 (required by Audiveris) | ~200 MB | GPL-2 + Classpath Exception |
| `downloads/` | Cached installer archives | varies | — |

Total: ~2 GB on disk.

## Install

```powershell
# From the repository root:
.\scripts\install_tools.ps1
```

If the script doesn't exist yet, install manually:

### 1. Python 3.11 embeddable

Download the Windows embeddable zip from <https://www.python.org/downloads/release/python-3119/> and extract into `tools/python/`.

### 2. ffmpeg

Download a Windows static build from <https://www.gyan.dev/ffmpeg/builds/> (essentials build is enough). Extract so that `tools/ffmpeg/bin/ffmpeg.exe` exists.

### 3. Eclipse Temurin JDK 21

Download from <https://adoptium.net/temurin/releases/?version=21&os=windows>. Extract so that `tools/jre/bin/java.exe` exists.

### 4. Audiveris 5.6.2

Download from <https://github.com/Audiveris/audiveris/releases/tag/5.6.2>. Extract so that `tools/audiveris/bin/Audiveris.bat` and `tools/audiveris/lib/audiveris.jar` exist.

## Why bundled (not system PATH)?

Reproducibility. Audiveris in particular is sensitive to JDK version, and we want a fresh checkout to produce identical results on any Windows machine without depending on whatever Python/Java the user has installed.

## License compliance

All four tools are open-source, but they are **not** redistributed inside this repository. They are downloaded directly from their official upstream releases at install time. See the top-level `NOTICE` file for the full attribution.
