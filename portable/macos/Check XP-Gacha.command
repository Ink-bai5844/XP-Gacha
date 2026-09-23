#!/bin/bash
set -u
cd -- "$(dirname -- "$0")" || exit 1
unset PYTHONHOME PYTHONPATH __PYVENV_LAUNCHER__

check_xp_gacha() {
    if [ "$(/usr/bin/uname -s)" != "Darwin" ] || [ "$(/usr/bin/uname -m)" != "arm64" ]; then
        printf '%s\n' 'This package requires native Apple Silicon macOS. Intel and Rosetta are unsupported.' >&2
        return 1
    fi
    macos_version=$(/usr/bin/sw_vers -productVersion)
    if [ "${macos_version%%.*}" -lt 15 ]; then
        printf '%s\n' 'This package requires macOS 15 or later.' >&2
        return 1
    fi
    if [ ! -x "runtime/python/bin/python3" ]; then
        printf '%s\n' 'Bundled Python is missing or not executable. Extract the complete release tar.gz first.' >&2
        return 1
    fi
    "runtime/python/bin/python3" -E -s -B portable_launcher.py doctor
}

check_xp_gacha
result=$?
if [ -t 0 ]; then
    read -r -p 'Check finished. Press Return to close this window. ' reply
fi
exit "$result"
