"""Label each generated continuation as rhyme / near_rhyme / repetition / none.

Implements Phase 3 steps 5 + 7 of the revision plan. Reads the attribution
graphs written by `generation-gemma-3-*.py`, reconstructs what each model
actually generated, finds the rhyme word, and scores it against the prompt's
target using the CMUdict layer already built in `tools/prompt_set.py`.

Emits `experiment/rhyme_labels.json`, which is the config source the tracing
stage reads in place of the old hardcoded `RHYME_TOKEN` / `RHYME_STEP`.

No model, no GPU, no network -- pure JSON parsing plus CMUdict.

    python experiment/rhyme_labels.py              # all sizes found on disk
    python experiment/rhyme_labels.py 1b 4b        # selected sizes
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))

from prompt_set import PRIMARY, RHYMING_PART, is_content_word  # noqa: E402

# attribute() was called on CHAT_PREFIX + prompt, so every graph's
# metadata.prompt carries this prefix; generation never saw it.
CHAT_PREFIX = "<bos><start_of_turn>user\n"

SIZES = ["270m", "1b", "4b"]
GRAPHS = Path(__file__).parent / "graphs"
PROMPT_SET_PATH = REPO / "tools" / "prompt_set.json"
OUT_PATH = Path(__file__).parent / "rhyme_labels.json"

STEP_RE = re.compile(r"step-(\d+)-(.*)\.json$")
# create_graph_files slugs are re.sub(r'[\\/:*?"<>|]', "_", tok.strip()).replace(" ", "_")
WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class Label:
    size: str
    slug: str
    continuation: str
    rhyme_word: str | None  # what the model produced (last content word)
    target_word: str | None  # what the prompt line ended on
    rhyme_token: str | None  # the exact token string, for tokenizer.encode()
    rhyme_step: int | None  # generation step that produced it
    prefix_before_rhyme: str  # continuation up to (not including) the rhyme token
    label: str
    single_token: bool | None  # False => downstream encode(...)[0] would truncate
    n_steps: int
    note: str = ""


def _lemmatizer():
    """WordNet if its corpus is installed, else None. Lemma matching is a
    refinement of the repetition rule, not a prerequisite -- exact match
    still catches the failure mode the reviewers named."""
    try:
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        lem.lemmatize("tests")
        return lem
    except Exception:
        return None


LEMMATIZER = _lemmatizer()


def same_lemma(a: str, b: str) -> bool:
    if LEMMATIZER is None:
        return False
    for pos in ("n", "v", "a"):
        if LEMMATIZER.lemmatize(a, pos) == LEMMATIZER.lemmatize(b, pos):
            return True
    return False


def step_files(slug_dir: Path) -> list[tuple[int, str, Path]]:
    """[(step_index, filename_token_slug, path)] sorted by step."""
    out = []
    for p in slug_dir.glob("step-*.json"):
        m = STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), m.group(2), p))
    return sorted(out)


def reconstruct(slug_dir: Path, prompt_text: str) -> tuple[str, list[int]]:
    """Rebuild the continuation and the char offset at which each step's token
    begins.

    Each step-i graph was attributed on CHAT_PREFIX + prompt + tokens[:i], so
    stripping the two known prefixes from metadata.prompt yields exactly
    tokens[:i] with original spacing. The final step's own token is not present
    in any graph and is recovered from its filename slug.
    """
    files = step_files(slug_dir)
    if not files:
        raise ValueError(f"no step files in {slug_dir}")

    prefixes: list[str] = []
    for _step, _tok, path in files:
        meta = json.loads(path.read_text()).get("metadata", {})
        full = meta.get("prompt", "")
        if not full.startswith(CHAT_PREFIX):
            raise ValueError(f"{path.name}: metadata.prompt lacks CHAT_PREFIX")
        rest = full[len(CHAT_PREFIX) :]
        if not rest.startswith(prompt_text):
            raise ValueError(f"{path.name}: metadata.prompt does not extend the prompt text")
        prefixes.append(rest[len(prompt_text) :])

    # prefixes[i] == tokens[:i], so step i's token begins at len(prefixes[i]).
    offsets = [len(p) for p in prefixes]
    continuation = prefixes[-1]

    # The final step's own token appears in no graph -- recover it from the
    # filename, where create_graph_files already stripped its leading space
    # and collapsed inner spaces to underscores.
    last_slug = files[-1][1].replace("_", " ").strip()
    if last_slug:
        sep = (
            " "
            if continuation and not continuation.endswith(" ") and last_slug[0].isalnum()
            else ""
        )
        offsets[-1] = len(continuation) + len(sep)
        continuation = continuation + sep + last_slug

    return continuation, offsets


def last_content_word(text: str) -> tuple[str, int] | None:
    """The final content word and its character offset.

    Line-final function words are skipped -- 'grab it' / 'stab it' must resolve
    to 'stab', or the flagship case scores as repetition on 'it'/'it'.
    """
    matches = list(WORD_RE.finditer(text))
    for m in reversed(matches):
        w = m.group(0)
        if is_content_word(w.lower()):
            return w.lower(), m.start()
    return (matches[-1].group(0).lower(), matches[-1].start()) if matches else None


def classify(generated: str, target: str) -> str:
    """Pre-registered rules (action_plan.md Phase 3 step 7). Repetition is
    checked first: a word trivially rhymes with itself."""
    if generated == target or same_lemma(generated, target):
        return "repetition"
    if generated not in PRIMARY or target not in PRIMARY:
        return "oov"
    ra, rb = RHYMING_PART.get(generated), RHYMING_PART.get(target)
    if ra is None or rb is None:
        return "no_primary_stress"
    if ra == rb:
        return "rhyme"
    if ra.split()[0] == rb.split()[0]:
        return "near_rhyme"
    return "none"


def label_one(size: str, slug: str, slug_dir: Path, record: dict) -> Label:
    prompt_text = record["prompt_text"]
    n_steps = len(step_files(slug_dir))

    # Derive the target with the SAME last-content-word rule used on the
    # generated line. prompt_set.json's `rhyme_word` is the literal final word,
    # which is a function word for the `original` anchor ("grab it," -> "it").
    # Comparing against "it" would score the flagship stab/grab case as `none`,
    # which is exactly what the last-content-word rule exists to prevent.
    target_found = last_content_word(prompt_text)
    target = target_found[0] if target_found else None
    declared = (record.get("rhyme_word") or "").lower() or None
    target_note = ""
    if declared and target and declared != target:
        target_note = f"target from last content word ({target}); prompt_set rhyme_word={declared}"

    try:
        continuation, offsets = reconstruct(slug_dir, prompt_text)
    except ValueError as e:
        return Label(
            size=size,
            slug=slug,
            continuation="",
            rhyme_word=None,
            target_word=target,
            rhyme_token=None,
            rhyme_step=None,
            prefix_before_rhyme="",
            label="degenerate",
            single_token=None,
            n_steps=n_steps,
            note=str(e),
        )

    found = last_content_word(continuation)
    if found is None:
        return Label(
            size=size,
            slug=slug,
            continuation=continuation,
            rhyme_word=None,
            target_word=target,
            rhyme_token=None,
            rhyme_step=None,
            prefix_before_rhyme="",
            label="degenerate",
            single_token=None,
            n_steps=n_steps,
            note="no word in continuation",
        )
    word, pos = found

    # Which step emitted it: the last step whose token starts at or before pos.
    step = max((i for i, off in enumerate(offsets) if off <= pos), default=0)
    # Single-token iff no later step begins inside the word.
    end = pos + len(word)
    single = not any(off > pos and off < end for off in offsets)

    label = classify(word, target) if target else "unscoreable"
    note = target_note if target else "no content word in prompt line"
    if target and not single:
        note = ("multi-token rhyme word: downstream encode(...)[0] would truncate. " + note).strip()

    return Label(
        size=size,
        slug=slug,
        continuation=continuation,
        rhyme_word=word,
        target_word=target,
        # The token string as generated, leading space included -- this is what
        # gets tokenizer.encode()'d downstream, so the space matters.
        rhyme_token=continuation[offsets[step] : end],
        rhyme_step=step,
        # Everything the model had seen when it chose the rhyme word; the
        # intervention stage uses prompt_text + this as its measurement prompt.
        prefix_before_rhyme=continuation[: offsets[step]],
        label=label,
        single_token=single,
        n_steps=n_steps,
        note=note,
    )


def main(sizes: list[str]) -> None:
    prompt_set = {r["slug"]: r for r in json.loads(PROMPT_SET_PATH.read_text())}
    results: list[Label] = []

    for size in sizes:
        root = GRAPHS / f"gemma-3-{size}-it"
        if not root.is_dir():
            print(f"[{size}] no graph dir at {root} -- skipping")
            continue
        for slug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            slug = slug_dir.name
            if slug not in prompt_set:
                print(f"[{size}] {slug}: not in prompt_set.json -- skipping")
                continue
            results.append(label_one(size, slug, slug_dir, prompt_set[slug]))

    for size in sizes:
        rows = [r for r in results if r.size == size]
        if not rows:
            continue
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.label] = counts.get(r.label, 0) + 1
        n_rhyme = counts.get("rhyme", 0)
        print(f"\n=== {size}: {n_rhyme}/{len(rows)} rhyme   {counts}")
        for r in sorted(rows, key=lambda x: x.slug):
            flag = "" if r.single_token is not False else "  [MULTI-TOKEN]"
            print(
                f"  {r.slug:10s} {r.label:12s} {str(r.rhyme_word):12s} vs "
                f"{str(r.target_word):10s} step={r.rhyme_step}{flag}"
            )
            if r.note:
                print(f"       note: {r.note}")

    OUT_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nwrote {OUT_PATH} ({len(results)} records)")

    if LEMMATIZER is None:
        print(
            "note: WordNet corpus absent -- repetition uses exact match only "
            "(python -m nltk.downloader wordnet to enable lemma matching)"
        )


if __name__ == "__main__":
    main([s for s in sys.argv[1:] if s in SIZES] or SIZES)
