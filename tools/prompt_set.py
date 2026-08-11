"""Phase 1 (word-first) prompt-set selection tool.

Implements action_plan.md Phase 1 steps 0-5, following PeRDict's actual
method (Crossley et al. 2024) of restricting rhyme-family-size to a
frequency-banded word list rather than raw CMUdict, so obscure/proper-noun
rhymes (e.g. "pixel" <-> "bichsel") don't pollute the "hard" band. PeRDict
used COCA; we use `wordfreq` as a pip-installable, cleanly-licensed stand-in
(its frozen ~2021 snapshot is a plausible feature here, not a bug: it
predates LLM-generated text polluting frequency norms).

Rhyme matching uses each word's PRIMARY CMUdict pronunciation only, not the
`pronouncing` library's default of unioning across all listed variants.
Raw CMUdict marks heteronyms/alternate pronunciations as separate entries
(e.g. "tear" / "tear(2)", "again" / "again(2)") - `pronouncing.rhymes()`
strips the "(N)" suffix and merges them, which silently unions phonetically
distinct rhyme families under one word (e.g. "tear" the verb, T-EH-R,
rhymes with "care"/"fair"; "tear" the noun, T-IH-R, rhymes with
"clear"/"near" - a heteronym, not a pronunciation variant of one word/
meaning; "again" similarly merges a dominant AH0-G-EH1-N reading that
rhymes with "when" with a rarer AH0-G-EY1-N reading that rhymes with
"brain", producing rhyme pairs that only work under the less common
reading). `load_primary_pronunciations()` parses CMUdict's raw entries
directly and keeps only the unsuffixed (primary) line per word.

Two metrics are computed:
  - `family_size`: raw CMUdict rhyme count (Phase 1 step 0, unrestricted).
    Kept for sanity-checking and as the unrestricted reference point.
  - `banded_family_size`: rhyme count restricted to the top-N wordfreq band
    (this module's primary selection metric).

Band size is pinned to 10,000: at 1,000/2,500/5,000 the fraction of
zero-in-band-rhyme words (~34-41%) exceeds the tertile's 33rd-percentile
mark, so the "hard" cutoff degenerates to 0 (indistinguishable from "no
rhyme at all"). At 10,000 the zero-rhyme fraction drops to ~29%, under the
tertile threshold, giving a non-degenerate hard band. PeRDict's own
headline models used the 1,000/2,500 bands, but their Table 2 reports the
same zero-inflation pattern (e.g. only 538/994 words with any rhyme at the
1,000-word band) - the wider band is a deviation from their exact choice
of band, not from their method.

Run directly to print the shortlists; final word picks and sentences are
decided by hand and hardcoded into `tools/prompt_set.json`.
"""

import random
from collections import Counter

import cmudict
import nltk
import pronouncing
from wordfreq import top_n_list

MIN_WORD_LEN = 3
MAX_WORD_LEN = 8
LOWER_PERCENTILE = 33
UPPER_PERCENTILE = 67
SHORTLIST_SIZE = 30
SHORTLIST_SEED = 0
FREQUENCY_BAND_SIZE = 10_000
MAX_PER_FAMILY = 2

CONTENT_TAGS = {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS"}
FUNCTION_TAGS = {"PRP", "PRP$", "IN", "DT"}


def load_primary_pronunciations() -> dict[str, str]:
    """CMUdict entries, keeping only each word's primary (unsuffixed) line.

    Raw CMUdict lines look like "tear T EH1 R" and "tear(2) T IH1 R" - the
    "(N)" suffix marks alternate pronunciations/heteronyms as distinct
    entries. Skipping suffixed entries gives one deterministic pronunciation
    per word, avoiding the variant-merging problem described in the module
    docstring.
    """
    primary: dict[str, str] = {}
    for line in cmudict.raw().splitlines():
        if not line or line.startswith(";;;"):
            continue
        word, phones = line.split(" ", 1)
        if "(" in word:
            continue
        primary[word.lower()] = phones
    return primary


PRIMARY = load_primary_pronunciations()
RHYMING_PART = {w: pronouncing.rhyming_part(p) for w, p in PRIMARY.items()}


def family_size(word: str) -> int:
    """Raw CMUdict rhyme-family size per action_plan.md Phase 1 step 0,
    computed from each word's primary pronunciation only (see module
    docstring)."""
    word = word.lower()
    if word not in PRIMARY:
        raise ValueError(f"{word!r} not found in CMUdict")
    rp = RHYMING_PART[word]
    return sum(1 for w, other_rp in RHYMING_PART.items() if w != word and other_rp == rp)


def reference_word_list() -> list[str]:
    return sorted(w for w in PRIMARY if w.isalpha() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN)


def is_primary_pronunciation_acronym(word: str) -> bool:
    """True if CMUdict's primary pronunciation for `word` spells it out
    letter-by-letter (syllable count == letter count), e.g. "dna" ->
    D-IY-EH-N-EY, 3 syllables for 3 letters. Real words that merely have a
    *secondary* abbreviation reading (e.g. "cod", "corp", "ins") aren't
    caught by this, since only the primary entry is checked - consistent
    with using primary-only pronunciations throughout this module."""
    return pronouncing.syllable_count(PRIMARY[word]) == len(word)


def frequency_band(n: int = FREQUENCY_BAND_SIZE) -> set[str]:
    """Top-n wordfreq words, restricted to CMUdict-pronounceable words in the
    same length/alpha range as `reference_word_list`, excluding words whose
    primary CMUdict pronunciation is an acronym spelling (see
    `is_primary_pronunciation_acronym`). Residual acronym leakage from words
    with a real primary pronunciation but a secondary acronym reading (e.g.
    "sri", "sci") isn't caught here - left for manual shortlist review."""
    return {
        w
        for w in top_n_list("en", n)
        if w.isalpha()
        and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN
        and w in PRIMARY
        and not is_primary_pronunciation_acronym(w)
    }


def banded_family_size_all(band: set[str]) -> dict[str, int]:
    """Rhyme-family size for every word in `band`, restricted to rhymes
    within `band` (primary pronunciations only), per PeRDict's frequency-band
    restriction: recompute rhyme counts within the reduced word list, not
    just filter candidates after the fact. Grouped by rhyming_part instead of
    pairwise comparison for O(n) rather than O(n^2)."""
    rp_of = {w: RHYMING_PART[w] for w in band}
    counts = Counter(rp_of.values())
    return {w: counts[rp] - 1 for w, rp in rp_of.items()}


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        raise ValueError("empty distribution")
    k = (len(sorted_values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def build_reference_distribution() -> dict[str, int]:
    words = reference_word_list()
    return banded_family_size_all(set(words))


def build_banded_distribution(band: set[str]) -> tuple[dict[str, int], float, float]:
    sizes = banded_family_size_all(band)
    sorted_sizes = sorted(sizes.values())
    low_cut = percentile(sorted_sizes, LOWER_PERCENTILE)
    high_cut = percentile(sorted_sizes, UPPER_PERCENTILE)
    return sizes, low_cut, high_cut


def sample_within_ties(
    candidates: list[tuple[str, int]], n: int, reverse: bool = False
) -> list[tuple[str, int]]:
    """Sort by family_size, but shuffle within each tied family_size so large
    tie pools don't just surface in alphabetical order.

    `candidates` is sorted by word first: it's typically built from a dict
    keyed off a set (`frequency_band()`), whose iteration order is randomized
    per-process by Python's string hash randomization, so shuffling it
    directly would make the output non-reproducible across runs even with a
    fixed seed. Sorting first pins a stable starting order for the seed to
    act on.
    """
    rng = random.Random(SHORTLIST_SEED)
    shuffled = sorted(candidates, key=lambda item: item[0])
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: item[1], reverse=reverse)
    return shuffled[:n]


def diversify_by_family(
    candidates: list[tuple[str, int]], n: int, cap: int = MAX_PER_FAMILY, reverse: bool = False
) -> list[tuple[str, int]]:
    """Cap how many words from the same rhyming_part can appear, so a
    shortlist doesn't collapse into one dominant rhyme cluster wearing many
    different words (e.g. the -ay/-eigh family swallowing the entire "easy"
    band, since word choice within a tied family_size is otherwise
    arbitrary). Groups are ordered deterministically (sorted, then
    seed-shuffled, then sorted by the group's family_size) and `cap` words
    are taken from each via `sample_within_ties` before moving to the next
    group, so groups are exhausted in family-size order rather than in an
    arbitrary interleaving.
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for w, size in candidates:
        groups.setdefault(RHYMING_PART[w], []).append((w, size))

    group_items = sorted(groups.items(), key=lambda kv: kv[0])
    rng = random.Random(SHORTLIST_SEED)
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: kv[1][0][1], reverse=reverse)

    result: list[tuple[str, int]] = []
    for _, members in group_items:
        result.extend(sample_within_ties(members, cap, reverse=reverse))
        if len(result) >= n:
            break
    return result[:n]


def shortlist_bands(
    sizes: dict[str, int], low_cut: float, high_cut: float
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Hard band: 0 < family_size <= low_cut. Easy band: family_size >= high_cut.
    Median band: low_cut < family_size < high_cut (strictly between the two cutoffs,
    so it doesn't overlap hard/easy) - added to support a graded difficulty
    comparison (hard/median/easy) rather than just a binary hard-vs-easy contrast,
    per reviewer feedback (D2/W2) asking for evidence distinguishing genuine
    planning from a two-point artifact."""
    hard_pool = [(w, n) for w, n in sizes.items() if 0 < n <= low_cut]
    easy_pool = [(w, n) for w, n in sizes.items() if n >= high_cut]
    median_pool = [(w, n) for w, n in sizes.items() if low_cut < n < high_cut]
    hard = diversify_by_family(hard_pool, SHORTLIST_SIZE)
    easy = diversify_by_family(easy_pool, SHORTLIST_SIZE, reverse=True)
    median = diversify_by_family(median_pool, SHORTLIST_SIZE)
    return hard, easy, median


def is_content_word(word: str) -> bool:
    tag = nltk.pos_tag([word])[0][1]
    return tag in CONTENT_TAGS


def repetition_control_candidates(sizes: dict[str, int]) -> list[tuple[str, int, str]]:
    """POS-filtered candidates per Phase 1 step 3, exempt from the easy/hard axis.

    Diversified by rhyming_part first (see `diversify_by_family`) for the
    same reason as the easy/hard shortlists: without it, this pool is drawn
    from the same skewed distribution and collapses into one rhyme cluster.
    """
    pool = diversify_by_family(list(sizes.items()), n=len(sizes), reverse=True)
    out = []
    for word, size in pool:
        tag = nltk.pos_tag([word])[0][1]
        if tag in CONTENT_TAGS:
            out.append((word, size, tag))
        if len(out) >= SHORTLIST_SIZE:
            break
    return out


def print_distribution_summary(label: str, sizes: dict[str, int]) -> None:
    sorted_sizes = sorted(sizes.values())
    zero = sum(1 for n in sorted_sizes if n == 0)
    print(f"{label}: {len(sizes)} words, {zero} zero-rhyme ({100 * zero / len(sizes):.1f}%)")
    for p in (10, 25, 33, 50, 67, 75, 90):
        print(f"  {p}th percentile: {percentile(sorted_sizes, p):.1f}")


def main() -> None:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    print(
        "Building raw CMUdict reference distribution (unrestricted, primary pronunciation only)..."
    )
    raw_sizes = build_reference_distribution()
    print_distribution_summary("Raw CMUdict", raw_sizes)

    print(f"\nBuilding wordfreq top-{FREQUENCY_BAND_SIZE} banded distribution...")
    band = frequency_band()
    banded_sizes, low_cut, high_cut = build_banded_distribution(band)
    print_distribution_summary("Banded", banded_sizes)
    print(
        f"Tertile cutoffs (banded) -> hard: 0 < family_size <= {low_cut:.1f}, "
        f"easy: family_size >= {high_cut:.1f}"
    )

    hard, easy, median = shortlist_bands(banded_sizes, low_cut, high_cut)
    print(f"\nHard shortlist (bottom third, banded), {len(hard)} words:")
    print(", ".join(f"{w}({n})" for w, n in hard))
    print(f"\nEasy shortlist (top third, banded), {len(easy)} words:")
    print(", ".join(f"{w}({n})" for w, n in easy))
    print(f"\nMedian shortlist (middle third, banded), {len(median)} words:")
    print(", ".join(f"{w}({n})" for w, n in median))

    print("\nRepetition-control candidates (content-word POS tags, high banded family size):")
    control = repetition_control_candidates(banded_sizes)
    print(", ".join(f"{w}({n},{tag})" for w, n, tag in control))

    # Sanity checks: "it" resolves via its primary pronunciation; "tear" and
    # "again" no longer inherit a rhyme family from their secondary
    # pronunciation (see module docstring).
    print(f"\nSanity check - 'it' raw family_size: {family_size('it')}")
    tear_rhymes = [w for w in PRIMARY if w != "tear" and RHYMING_PART[w] == RHYMING_PART["tear"]]
    again_rhymes = [w for w in PRIMARY if w != "again" and RHYMING_PART[w] == RHYMING_PART["again"]]
    print(f"Sanity check - 'tear' rhymes (primary pronunciation only): {tear_rhymes[:10]}")
    print(f"Sanity check - 'again' rhymes (primary pronunciation only): {again_rhymes[:10]}")


if __name__ == "__main__":
    main()
