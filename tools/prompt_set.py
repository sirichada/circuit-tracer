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
from wordfreq import top_n_list, zipf_frequency

MIN_WORD_LEN = 3
MAX_WORD_LEN = 8
LOWER_PERCENTILE = 33
UPPER_PERCENTILE = 67
SHORTLIST_SIZE = 30
SHORTLIST_SEED = 0
FREQUENCY_BAND_SIZE = 10_000

# One word per rhyme family in a shortlist. Was 2; tightened because the easy
# band has only ~27 distinct families, so a 5-prompt selection drawn with cap=2
# risks being five sentences on two rhyme sounds.
MAX_PER_FAMILY = 1

# Difficulty bands are ABSOLUTE rhyme counts, not percentile cutoffs.
#
# Tertiles were tried and abandoned. Under the primary-stress rule the
# fraction of band words with zero in-band rhymes rises to 35.8%, above the
# 33rd percentile, so the hard cutoff lands on 0 - i.e. the hard band becomes
# "words with no rhyme at all", which a couplet cannot use. Restoring a
# workable tertile needs a 20,000-word band, which departs further from
# PeRDict's published bands (largest: 10,000) and pushes "creator" out of the
# medium band. Absolute thresholds dissolve the problem and let the band stay
# at 10,000.
#
# The 10-19 gap between MEDIUM_MAX and EASY_MIN is deliberate, so medium and
# easy are separated rather than merely adjacent.
HARD_RAW_MAX = 1  # counted over ALL of CMUdict - see shortlist_bands
MEDIUM_MIN, MEDIUM_MAX = 2, 9  # counted within the frequency band
EASY_MIN = 20  # counted within the frequency band

# A medium word must ALSO not have more rhymes in total than an easy word has
# usable ones. This is a validity check, not a second difficulty axis: without
# it, "creator" qualifies as medium on 2 in-band rhymes while actually having
# 53 in total, its two in-band options being "later" and "greater" - among the
# most common words in English. That is the same band artifact that disqualifies
# "aaron" (56 raw) from the hard band.
#
# Deliberately tied to EASY_MIN rather than given its own number, so no new
# free parameter is introduced. Candidate supply does not constrain the choice
# (a cap of 10 still leaves 76 rhyme families, a cap of 40 leaves 375), which
# is exactly why the value must come from a reason rather than from its effect.
MEDIUM_ALL_MAX = EASY_MIN

# A hard word's single rhyme must itself be usable, or the prompt is
# impossible rather than hard. zipf 3.0 is roughly "occurs in ordinary text".
HARD_PARTNER_MIN_ZIPF = 3.0

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


def rhyming_part_primary(phones: str) -> str | None:
    """Rime measured from the PRIMARY-stressed vowel (CMUdict stress digit 1)
    to the end of the word. `None` when the word has no primary stress.

    `pronouncing.rhyming_part()` measures from the *last* stressed vowel,
    counting secondary stress (digit 2). The two differ for words carrying
    secondary stress after the primary one: "anyway" (EH1 N IY0 W EY2) gets
    rime "EY2" under `pronouncing`, but "EH1 N IY0 W EY2" here.

    PeRDict measures from the primary vowel, verified two independent ways
    against its published database (Crossley & Choi 2024):
      - Recomputing their rhyme counts over their own 47,865-word pool
        reproduces the published numbers for 93.5% of words under this rule,
        vs 70.5% under `pronouncing`'s last-stressed rule.
      - p. 784 reports 20 words dropped for "lacking stressed vowels",
        naming y'all, marketers, greedier, priciest. All four *do* carry
        secondary stress; what they lack is a primary one - and all four are
        absent from the published database. Words with no digit-1 stress are
        therefore excluded here too, mirroring that.
    """
    parts = phones.split()
    stressed = [i for i, p in enumerate(parts) if p.endswith("1")]
    return " ".join(parts[stressed[-1] :]) if stressed else None


PRIMARY = load_primary_pronunciations()
RHYMING_PART = {
    w: rime for w, p in PRIMARY.items() if (rime := rhyming_part_primary(p)) is not None
}

# Rhyme partners are counted over alphabetic entries only. Raw CMUdict
# includes hyphenated, possessive and punctuated forms ("back-up", "on-line",
# "politics'", "davis'"), which are the *same word* rather than rhymes of it -
# counting them inflates a word's apparent rhyme family and, worse, makes a
# word look like it has exactly one "rhyme" when that rhyme is itself
# (e.g. "online"/"on-line", "backup"/"back-up"). That directly corrupts the
# hard band, which is defined by genuine scarcity.
#
# Precedent: PeRDict restricted its pool to CMUdict INTERSECT ELP (Crossley &
# Choi 2024, p. 782), which excludes such entries. `reference_word_list()`
# below already applies `.isalpha()`, but that filter never reached the
# partner pool used for counting - this is a bug fix, not a new heuristic.
#
# Not caught here, and left to the manual naturalness pass (Phase 1 step 2)
# for the same reason proper nouns are: same-word spelling variants
# ("theater"/"theatre", "realise"/"realize") and abbreviations ("feb"/
# "february") are alphabetic, so no mechanical rule separates them from real
# rhymes without inventing an unvalidated threshold.
RHYME_POOL = {w: rp for w, rp in RHYMING_PART.items() if w.isalpha()}

# rhyming_part -> how many pool words share it. Lets `family_size` be O(1)
# instead of a scan over ~117k entries per call, which matters because the
# band selectors call it once per candidate word.
_POOL_COUNTS = Counter(RHYME_POOL.values())


def family_size(word: str) -> int:
    """Raw CMUdict rhyme-family size per action_plan.md Phase 1 step 0,
    computed from each word's primary pronunciation only (see module
    docstring), counting partners over `RHYME_POOL` rather than all of
    CMUdict."""
    word = word.lower()
    if word not in PRIMARY:
        raise ValueError(f"{word!r} not found in CMUdict")
    if word not in RHYMING_PART:
        raise ValueError(f"{word!r} has no primary-stressed vowel in CMUdict")
    rp = RHYMING_PART[word]
    return _POOL_COUNTS[rp] - (1 if word in RHYME_POOL else 0)


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
    rp_of = {w: RHYMING_PART[w] for w in band if w in RHYMING_PART}
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


def hard_pool(band: set[str]) -> list[tuple[str, int]]:
    """Hard band, defined on RAW CMUdict scarcity rather than in-band count.

    This asymmetry with medium/easy is the whole point. Defining hard as "1
    in-band rhyme" yields 864 words, nearly all band artifacts: "aaron" has 56
    raw CMUdict rhymes, "annie" 49, "antenna" 44, "allows" 40. Their in-band
    count is 1 only because the frequency band hides the rest - and the model
    knows those words regardless, so they are not hard to rhyme.

    action_plan.md reached this conclusion once already, rejecting "running"
    (in-band 1, raw 9) for "article" (raw 1): "band-restricted scarcity
    doesn't necessarily reflect true rhyme difficulty, and the hard band
    should reflect the latter."

    Medium and easy don't need this. Easy at in-band >=20 implies raw >=20, so
    no artifact is possible; medium is exposed only in the harmless direction.

    Words are still required to be in `band`, for naturalness - a scarce rhyme
    on a word nobody uses makes an odd prompt. The partner must clear
    HARD_PARTNER_MIN_ZIPF so the couplet is completable.
    """
    out = []
    for word in band:
        if word not in RHYMING_PART:
            continue
        if family_size(word) != HARD_RAW_MAX:
            continue
        rp = RHYMING_PART[word]
        partners = [w for w, r in RHYME_POOL.items() if r == rp and w != word]
        if not partners:
            continue
        if zipf_frequency(partners[0], "en") < HARD_PARTNER_MIN_ZIPF:
            continue
        out.append((word, family_size(word)))
    return out


def shortlist_bands(
    sizes: dict[str, int], band: set[str]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Shortlists for the three difficulty bands.

    `sizes` holds in-band rhyme counts (used for medium and easy); hard is
    computed separately from raw counts (see `hard_pool`). The three bands are
    a graded difficulty axis rather than a binary hard-vs-easy contrast, per
    reviewer feedback (D2/W2) asking for evidence distinguishing genuine
    planning from a two-point artifact.
    """
    medium_pool = [
        (w, n)
        for w, n in sizes.items()
        if MEDIUM_MIN <= n <= MEDIUM_MAX and family_size(w) <= MEDIUM_ALL_MAX
    ]
    easy_pool = [(w, n) for w, n in sizes.items() if n >= EASY_MIN]
    hard = diversify_by_family(hard_pool(band), SHORTLIST_SIZE)
    easy = diversify_by_family(easy_pool, SHORTLIST_SIZE, reverse=True)
    medium = diversify_by_family(medium_pool, SHORTLIST_SIZE)
    return hard, easy, medium


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
    banded_sizes, _, _ = build_banded_distribution(band)
    print_distribution_summary("Banded", banded_sizes)
    print(
        f"Absolute bands -> hard: raw CMUdict rhymes == {HARD_RAW_MAX} "
        f"(partner zipf >= {HARD_PARTNER_MIN_ZIPF}); "
        f"medium: {MEDIUM_MIN}-{MEDIUM_MAX} in-band and <= {MEDIUM_ALL_MAX} raw; "
        f"easy: >= {EASY_MIN} in-band (top decile of the band)"
    )

    hard, easy, medium = shortlist_bands(banded_sizes, band)
    print(f"\nHard shortlist (raw scarcity), {len(hard)} words - shown as word(raw):")
    print(", ".join(f"{w}({n})" for w, n in hard))
    print(f"\nEasy shortlist (in-band >= {EASY_MIN}), {len(easy)} words:")
    print(", ".join(f"{w}({n})" for w, n in easy))
    print(
        f"\nMedium shortlist (in-band {MEDIUM_MIN}-{MEDIUM_MAX}, "
        f"raw <= {MEDIUM_ALL_MAX}), {len(medium)} words:"
    )
    print(", ".join(f"{w}({n})" for w, n in medium))

    print("\nRepetition-control candidates (content-word POS tags, high banded family size):")
    control = repetition_control_candidates(banded_sizes)
    print(", ".join(f"{w}({n},{tag})" for w, n, tag in control))

    # Sanity checks: "it" resolves via its primary pronunciation; "tear" and
    # "again" no longer inherit a rhyme family from their secondary
    # pronunciation (see module docstring).
    print(f"\nSanity check - 'it' raw family_size: {family_size('it')}")
    tear_rhymes = [w for w in RHYME_POOL if w != "tear" and RHYME_POOL[w] == RHYMING_PART["tear"]]
    again_rhymes = [
        w for w in RHYME_POOL if w != "again" and RHYME_POOL[w] == RHYMING_PART["again"]
    ]
    print(f"Sanity check - 'tear' rhymes (primary pronunciation only): {tear_rhymes[:10]}")
    print(f"Sanity check - 'again' rhymes (primary pronunciation only): {again_rhymes[:10]}")


if __name__ == "__main__":
    main()
