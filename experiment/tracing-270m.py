"""Circuit tracing for Gemma-3-270M-it.

All logic lives in `tracing.py`; this file is config only.

    python experiment/tracing-270m.py
"""

from tracing import main

if __name__ == "__main__":
    main("270m")
