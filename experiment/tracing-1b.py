"""Circuit tracing for Gemma-3-1B-it.

All logic lives in `tracing.py`; this file is config only. 1B rhymes on 2 of
11 prompts (`realm`->helm, `beneath`->wreath), both of which 4B also rhymes --
the two clean-comparison prompts for the cross-scale claim.

    python experiment/tracing-1b.py
"""

from tracing import main

if __name__ == "__main__":
    main("1b")
