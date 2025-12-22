"""Allow running as: python -m discover <command> [options]"""

from .cli import main

if __name__ == '__main__':
    raise SystemExit(main())

