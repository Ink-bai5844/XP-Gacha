# Third-party runtime notices (macOS)

This portable distribution includes XP-Gacha and third-party runtimes and Python packages. Each component remains subject to its own license. Exact installed package versions are recorded in `requirements-lock.txt`; runtime versions and provenance are recorded in `BUILD-INFO.json` and the build runtime manifest.

- CPython 3.12.14 is distributed under the Python Software Foundation License. The relocatable macOS build is supplied by the [python-build-standalone project](https://github.com/astral-sh/python-build-standalone). Its original runtime files and bundled license notices are retained under `runtime/python`; licenses for libraries included by that distribution also apply.
- MySQL Community Server 8.4.11 is distributed by Oracle under GPLv2 and its applicable additional licensing terms. The original macOS runtime, including its license and documentation files, is retained under `runtime/mysql`. Consult those files for the exact terms and the [MySQL Community Downloads](https://dev.mysql.com/downloads/mysql/) for the corresponding source distribution.
- Python package license metadata and bundled resources are retained under `runtime/python/lib/python3.12/site-packages`, including package `.dist-info` directories. PyTorch, NumPy, SciPy and other packages retain their own license terms.
- Frontend dependencies remain under their respective licenses. The package includes the React/Vite production build generated from the repository's dependency lockfile.

No Microsoft Visual C++ runtime or Windows MySQL runtime is included in this macOS package. This application bundle has not been signed or notarized by Apple.
