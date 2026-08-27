"""Circuit tracing for Gemma-3-270M-it.

All logic lives in `tracing.py`; this file is config only. 270M produces no
genuine rhymes on the 11-prompt set, so it is the scaling floor / negative
case -- its prompts are still analysed (labelled `none` / `repetition`) so the
absence of planning features is measured rather than assumed.

    python experiment/tracing-270m.py --no-interventions
"""

from tracing import main

if __name__ == "__main__":
    main("270m")
