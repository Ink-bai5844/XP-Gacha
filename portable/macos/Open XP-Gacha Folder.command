#!/bin/bash
set -u
cd -- "$(dirname -- "$0")" || exit 1
/usr/bin/open "$PWD"
result=$?
if [ "$result" -ne 0 ] && [ -t 0 ]; then
    read -r -p 'Could not open the package folder. Press Return to close this window. ' reply
fi
exit "$result"
