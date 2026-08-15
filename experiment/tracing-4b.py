"""Circuit tracing for Gemma-3-4B-it.

All logic lives in `tracing.py`; this file is config only. 4B carries the
headline results -- it rhymes on 7 of 11 prompts (plus 2 near-rhymes), so it
is the only size with enough rhyme events to support a mechanistic claim.

    python experiment/tracing-4b.py
"""

from tracing import main

if __name__ == "__main__":
    main("4b")
