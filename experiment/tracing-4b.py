"""Circuit tracing for Gemma-3-4B-it.

All logic lives in `tracing.py`; this file is config only.

    python experiment/tracing-4b.py
"""

from tracing import main

if __name__ == "__main__":
    main("4b")
