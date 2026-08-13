"""Phase 3 steps 5 + 7: natural-match check and CMUdict rhyme labels.

Reconstructs each model's generated continuation from the attribution-graph
JSON written by `experiment/generation-gemma-3-*.py`, locates the rhyme word
and the step at which the model committed to it, labels the continuation
against the prompt's own line-ending word (rhyme / near-rhyme / repetition),
and buckets each prompt by whether the three model sizes converged on the same
rhyme target without any forcing.

Two reviewer objections motivate this (see `action_plan.md` Phase 3):

  - The "4B confound": each size free-generates its own continuation, so the
    sizes are traced on different rhyme words at different steps. Any
    cross-scale timing difference could come from the differing targets rather
    than from scale. Bucketing the prompts turns that from an unanswered flaw
    into a measured number.
  - "Repetition, not rhyme": the smaller models appear to echo the prompt's
    last word. The identical-word check makes that objective rather than
    eyeballed.

No model load, no GPU - this reads graph JSON and CMUdict only.

Rhyme unit is the **last content word**, skipping line-final function words.
The original prompt's line 1 ends "had to grab it," and 4B generates "had to
crap it." - comparing strictly the last word gives `it`/`it` -> repetition,
which is plainly wrong. Comparing across the word boundary ("grab it"/"crap
it") is more faithful to how the couplet actually rhymes, but makes the rhyme
step ambiguous (`crap` is step 8, `it` is step 9) and the planning-vs-execution
split in `tracing-*.py` is defined against that index. Last-content-word keeps
it unambiguous at step 8. The residual loss is recorded as `cross_word_note`
on that prompt rather than papered over.

Adverbs count as content words here: they are in neither `CONTENT_TAGS` nor
`FUNCTION_TAGS`, and skipping them would discard genuine line-final rhymes
(1B's repetition_control ends "...say it anyway.", where `anyway` really does
rhyme with `say`).

Run directly; writes `tools/natural_match.json` and prints the labels table.
"""

import json
import re
import string
from pathlib import Path

import nltk

from prompt_set import FUNCTION_TAGS, PRIMARY, RHYMING_PART, shares_onset

REPO_ROOT = Path(__file__).parent.parent
PROMPT_SET_PATH = Path(__file__).parent / "prompt_set.json"
GRAPHS_DIR = REPO_ROOT / "experiment" / "graphs"
OUTPUT_PATH = Path(__file__).parent / "natural_match.json"

SIZES = ["gemma-3-270m", "gemma-3-1b", "gemma-3-4b"]
STEP_RE = re.compile(r"step-(\d+)-(.+)\.json$")

# 'grab it' / 'crap it' rhymes across the word boundary; the single-word rime
# rule can only see 'grab' vs 'crap', whose rimes differ (AE1 B vs AE1 P).
# CMUdict marks vowel stress with a digit, and rimes are compared as exact
# strings - so 'say' (EY1) and 'anyway' (EY2) do not match despite sharing a
# vowel. Kept deliberately: `prompt_set.py`'s `family_size`, which selected the
# prompt set in Phase 1, compares the same way, and relaxing it here alone
# would put selection and labelling on subtly different notions of rhyme. The
# bias is conservative - it undercounts rhymes rather than overcounting.
STRESS_NOTE = (
    "Rimes are measured from the PRIMARY-stressed vowel and compared as exact "
    "CMUdict strings including stress digits, matching PeRDict (Crossley & Choi "
    "2024). Verified against PeRDict's published database: this rule reproduces "
    "their counts for 93.5% of their 47,865-word pool, vs 70.5% measuring from "
    "the last stressed vowel and ~40% ignoring stress. Consequence: 'anyway' "
    "(EH1 N IY0 W EY2) does not rhyme with 'say' (EY1) - under PeRDict's rule "
    "'anyway' has zero rhymes in the entire database. The bias is conservative: "
    "it undercounts rhymes rather than overcounting them."
)

ONSET_NOTE = (
    "Identical onsets are excluded: rime riche, which English prosody treats as "
    "a failed rhyme. Crossley & Choi 2024 state the same clause (p. 783), read "
    "literally on 'if present', so it cannot fire on a vowel-initial word. "
    "Note their released database appears not to apply it - our counts fit "
    "their published numbers better without it (93.5% vs 86.9%) - so counts "
    "here differ slightly from that database. The rule is adopted on the "
    "prosodic argument, not on reconstruction. It does NOT remove morphological "
    "pairs such as titled/entitled, whose onsets differ (T vs EH0 N T)."
)

CROSS_WORD_NOTE = (
    "'grab it'/'crap it' rhymes across the word boundary; the single-word rime "
    "rule compares 'grab' (AE1 B) against the generated rhyme word, so a "
    "boundary-spanning rhyme scores at best near_rhyme. family_size is null in "
    "prompt_set.json for the same reason."
)


def strip_punct(word: str) -> str:
    return word.strip(string.punctuation + string.whitespace)


def last_content_word(words: list[str]) -> int | None:
    """Index of the last content word, skipping trailing function words.

    Tags the sequence as a whole rather than word by word - tagging a bare word
    out of context is unreliable. Adverbs are in neither tag set and are kept
    (see module docstring).
    """
    if not words:
        return None
    tags = nltk.pos_tag(words)
    for i in range(len(words) - 1, -1, -1):
        if tags[i][1] not in FUNCTION_TAGS:
            return i
    return None


def baseline_word(prompt_text: str) -> str:
    """The word the prompt's line 1 rhymes on.

    Derived from `prompt_text` rather than read from the record's `rhyme_word`
    so the rule is uniform across all five prompts; the caller asserts the two
    agree. Applies the *same* last-content-word rule used on the generated
    side, so the original prompt's "...had to grab it," yields `grab` rather
    than the function word `it` - otherwise every model would trivially score
    `repetition` against `it`.
    """
    lines = [ln for ln in prompt_text.splitlines() if ln.strip()]
    words = [strip_punct(w).lower() for w in lines[-1].split()]
    words = [w for w in words if w]
    idx = last_content_word(words)
    assert idx is not None, f"no content word in prompt line: {lines[-1]!r}"
    return words[idx]


def load_steps(prompt_dir: Path) -> dict[int, Path]:
    steps = {}
    for f in prompt_dir.glob("step-*.json"):
        m = STEP_RE.search(f.name)
        if m:
            steps[int(m.group(1))] = f
    return steps


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("metadata", {})


def is_looped(tokens: list[str], min_repeats: int = 3) -> bool:
    """Detect a run that never terminated and cycled instead.

    270M's `easy` run repeated "he was late," until it OOMed at step 13, never
    emitting a stop token, so it has no well-defined rhyme word. Graphs alone
    can't tell us whether generation stopped cleanly - the stop token has no
    graph either way - so detect the cycle directly: any 3-gram occurring
    `min_repeats` times in a line this short is a loop, not natural repetition.
    """
    words = [t.strip().lower() for t in tokens if t.strip()]
    if len(words) < 3 * min_repeats:
        return False
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return any(grams.count(g) >= min_repeats for g in set(grams))


def reconstruct(prompt_dir: Path) -> tuple[list[str], str] | None:
    """Recover the generated tokens for one (size, slug) run.

    `generation-gemma-3-*.py` builds step i's input as `prompt_text +
    tokens_so_far`, so step i's graph holds tokens for steps 0..i-1 only. The
    prompt prefix length comes from step-00, whose graph is the bare prompt.
    The highest step's own token is recovered from its filename, since no graph
    is written for the step after it.
    """
    steps = load_steps(prompt_dir)
    if not steps or 0 not in steps:
        return None

    prefix_len = len(read_metadata(steps[0]).get("prompt_tokens") or [])
    if not prefix_len:
        return None

    top = max(steps)
    top_meta = read_metadata(steps[top])
    tokens = list(top_meta.get("prompt_tokens") or [])[prefix_len:]

    # The final step's own token exists only in its filename, whose slug was
    # `.strip().replace(" ", "_")`'d at write time - its leading space is
    # unrecoverable. Force it to start a new word rather than let it silently
    # glue onto the previous one ("to" + "say" -> "tosay"). Safe here: the
    # tokenizer check confirmed every rhyme word is a single token, so a
    # genuine sub-word piece never lands in final position.
    m = STEP_RE.search(steps[top].name)
    final = m.group(2).replace("_", " ")
    tokens.append(final)

    # Same stripped-space problem in the rendered text: restore a space before
    # the final token when it starts a word, so the continuation reads
    # correctly. Punctuation-leading finals ("." ",") attach as-is.
    joined = "".join(tokens[:-1])
    sep = " " if final[:1].isalnum() and joined[-1:].strip() else ""
    return tokens, joined + sep + final


def merge_words(
    tokens: list[str], boundary_at: int | None = None
) -> list[tuple[str, int, int]]:
    """Group tokens into whole words.

    Returns (word, first_step_index, n_alphanumeric_pieces). The piece count
    distinguishes genuine sub-word splitting from punctuation attaching to a
    word, which is not what the tokenizer check was about.

    Uses the leading-space convention preserved in `prompt_tokens` - a token
    beginning with a space starts a new word. This is why the graph metadata
    beats the filenames, whose slug is `.strip().replace(" ", "_")`'d and so is
    ambiguous about sub-word boundaries. `boundary_at` forces a word break at
    one index, for the filename-derived final token whose space was stripped.
    """
    words: list[tuple[str, int, int]] = []
    for i, tok in enumerate(tokens):
        if not tok.strip():
            continue
        alpha = 1 if any(c.isalnum() for c in tok) else 0
        if tok.startswith((" ", "\n")) or not words or i == boundary_at:
            words.append((tok.strip(), i, alpha))
        else:
            prev, start, n = words[-1]
            words[-1] = (prev + tok.strip(), start, n + alpha)
    return words


def find_rhyme(tokens: list[str]) -> tuple[str, int, bool] | None:
    """Walk backwards to the last content word.

    Returns (word, step index of its first token, whether sub-word merging was
    needed). Skips punctuation-only tokens and line-final function words;
    adverbs are kept (see module docstring).
    """
    words = merge_words(tokens, boundary_at=len(tokens) - 1)
    if not words:
        return None

    cleaned = [(strip_punct(w).lower(), i, n) for w, i, n in words]
    cleaned = [(w, i, n) for w, i, n in cleaned if w]
    if not cleaned:
        return None

    idx = last_content_word([w for w, _, _ in cleaned])
    if idx is None:
        return None

    word, step, n_pieces = cleaned[idx]
    # Only real sub-word splitting counts; a trailing "." merging onto a word
    # is not the case the tokenizer check was about.
    return word, step, n_pieces > 1


def label(generated: str, target: str) -> str:
    """Order matters: the identical-word exclusion fires before rhyme, since a
    word trivially rhymes with itself - which is exactly the repetition failure
    mode reviewers flagged. Rimes come from `prompt_set.RHYMING_PART`, measured
    from the primary-stressed vowel (see STRESS_NOTE / ONSET_NOTE)."""
    if generated == target:
        return "repetition"
    if generated not in RHYMING_PART or target not in RHYMING_PART:
        return "oov"
    if RHYMING_PART[generated] == RHYMING_PART[target]:
        # Identical onsets are rime riche, which English prosody treats as a
        # failed rhyme rather than a rhyme (see prompt_set.shares_onset).
        return "none" if shares_onset(generated, target) else "rhyme"
    gen_vowel = RHYMING_PART[generated].split()[0]
    tgt_vowel = RHYMING_PART[target].split()[0]
    if gen_vowel == tgt_vowel:
        return "near_rhyme"
    return "none"


def analyse_run(prompt_dir: Path, target: str) -> dict:
    result = reconstruct(prompt_dir)
    if result is None:
        return {"label": "degenerate", "reason": "no graphs or empty metadata"}

    tokens, continuation = result
    if is_looped(tokens):
        return {
            "label": "degenerate",
            "reason": "generation cycled without terminating; no well-defined rhyme word",
            "continuation": continuation,
            "n_steps": len(tokens),
        }

    found = find_rhyme(tokens)
    if found is None:
        return {
            "label": "degenerate",
            "reason": "no content word found",
            "continuation": continuation,
            "n_steps": len(tokens),
        }

    word, step, merged = found
    return {
        "rhyme_word": word,
        "rhyme_step": step,
        "n_steps": len(tokens),
        "label": label(word, target),
        "continuation": continuation,
        "truncated_tail": True,
        "subword_merged": merged,
    }


def main() -> None:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    prompt_set = json.loads(PROMPT_SET_PATH.read_text(encoding="utf-8"))
    prompts: dict[str, dict] = {}
    summary = {"clean": 0, "divergence": 0, "incomplete": 0}

    for record in prompt_set:
        slug = record["slug"]
        target = baseline_word(record["prompt_text"])
        assert target == record["rhyme_word"], (
            f"{slug}: prompt line ends on {target!r} but prompt_set.json records "
            f"rhyme_word={record['rhyme_word']!r} - these must agree"
        )

        sizes = {
            size: analyse_run(GRAPHS_DIR / size / slug, target) for size in SIZES
        }

        labels = [s["label"] for s in sizes.values()]
        words = [s.get("rhyme_word") for s in sizes.values()]
        steps = [s.get("rhyme_step") for s in sizes.values()]

        token_agreement = len(set(words)) == 1 and words[0] is not None
        step_agreement = len(set(steps)) == 1 and steps[0] is not None

        if "degenerate" in labels:
            bucket = "incomplete"
        elif token_agreement and "repetition" not in labels:
            bucket = "clean"
        else:
            bucket = "divergence"
        summary[bucket] += 1

        entry = {
            "bucket": bucket,
            "token_agreement": token_agreement,
            "step_agreement": step_agreement,
            "prompt_rhyme_word": target,
            "sizes": sizes,
        }
        if slug == "original":
            entry["cross_word_note"] = CROSS_WORD_NOTE
        prompts[slug] = entry

    output = {
        "generated_by": "tools/rhyme_labels.py",
        "rhyme_rule": "exact rime match measured from the primary-stressed vowel",
        "near_rhyme_rule": "primary stressed vowel matches, full rime differs",
        "onset_clause": ONSET_NOTE,
        "rhyme_unit": "last content word",
        "content_word_rule": "skip function words only; adverbs count",
        "lemma_check": "string_equality",
        "baseline_word": "last content word of the prompt's line 1",
        "stress_note": STRESS_NOTE,
        "prompts": prompts,
        "summary": summary,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"{'prompt':<20} {'size':<14} {'rhyme':<12} {'step':>4} {'label':<12} bucket")
    print("-" * 82)
    for slug, entry in prompts.items():
        for size, s in entry["sizes"].items():
            print(
                f"{slug:<20} {size:<14} {str(s.get('rhyme_word', '-')):<12} "
                f"{str(s.get('rhyme_step', '-')):>4} {s['label']:<12} {entry['bucket']}"
            )
        print()

    print(f"summary: {summary}")
    if any(s.get("subword_merged") for e in prompts.values() for s in e["sizes"].values()):
        print("NOTE: sub-word merging fired - the tokenizer check predicted it would not")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
