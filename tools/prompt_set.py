"""Phase 1 (word-first) prompt-set selection tool.

Follows PeRDict's actual
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

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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

PROMPT_SET_PATH = Path(__file__).parent / "prompt_set.json"
TOKENIZER_REPORT_PATH = Path(__file__).parent.parent / "experiment" / "tokenizer_check_report.json"

# google/gemma-3-{270m,1b-pt,4b-pt} are gated HF repos; --check-tokenizer
# requires `transformers` and a HF auth token with access.
TOKENIZER_MODELS = ["google/gemma-3-270m", "google/gemma-3-1b-pt", "google/gemma-3-4b-pt"]

# Evenly-spaced grid in log10(rhymes_all), 1 to 115: step = log10(115)/9 ~=
# 0.229 log10 units (~1.69x per step). Fixed in advance rather than found by
# iterative gap-patching, which has no natural stopping point - see
# Prompt set redesign: n=10, evenly log-spaced.
GRID_TARGETS = [1, 2, 3, 5, 8, 14, 24, 40, 68, 115]

# One word per rhyme family in a candidate list, so a list spans distinct
# rhyme sounds rather than one family wearing many spellings.
MAX_PER_FAMILY = 1

# Difficulty is a label read off `family_size`, not a band with tuned cutoffs:
# hard = 1-9, medium = 10-99, easy = 100+. It is the integer part of
# log10(family_size), which the continuous analysis already contains, so the
# labels introduce no parameter of their own. They are descriptive; the
# headline analysis regresses on log10(family_size) directly.
#
# Earlier revisions used percentile tertiles, then absolute in-band thresholds
# with a separate raw-count cap. Both were abandoned: they carried tuned
# constants that no claim rested on, and a short paper defending them hands
# reviewers surface to attack.
DIFFICULTY_LABELS = {0: "hard", 1: "medium", 2: "easy"}

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

ONSET = {
    w: PRIMARY[w][: len(PRIMARY[w]) - len(rime)].strip() for w, rime in RHYME_POOL.items()
}


def shares_onset(a: str, b: str) -> bool:
    """Whether the onset clause excludes this pair.

    Two words are not a rhyme if the consonants before the stressed vowel are
    identical - identical rhyme, or *rime riche*, which English prosody
    excludes on its own terms (French verse prizes it; English treats it as a
    failed rhyme). Crossley & Choi 2024 state the same clause for PeRDict
    (p. 783): "the consonant(s) before the stressed vowel (if present) had to
    differ". "If present" is read literally - a vowel-initial word has no
    onset, so the clause cannot fire on it.

    Practically this removes same-word variants that would otherwise look like
    scarce rhymes: feb/february, theater/theatre, realise/realize,
    favorite/favourite, licence/license, junior/jr.

    NOT removed, and a standing trap: morphological pairs like titled/entitled
    and counter/encounter. The onset is the whole pre-stress cluster, so
    "entitled" gives EH0 N T against "titled"'s T - they differ, and the pair
    survives. Those need the separate task-mechanism judgement in the manual
    pass, because a model can complete such a couplet by affixation rather
    than by phonological retrieval.
    """
    oa, ob = ONSET.get(a, ""), ONSET.get(b, "")
    return bool(oa and ob and oa == ob)


# rhyming_part -> Counter of onsets among pool words sharing it. Lets
# `family_size` be O(1) instead of scanning ~117k entries per call.
_ONSET_COUNTS: dict[str, Counter] = {}
for _w, _rp in RHYME_POOL.items():
    _ONSET_COUNTS.setdefault(_rp, Counter())[ONSET[_w]] += 1


def rhyme_partners(word: str) -> list[str]:
    """Every pool word that rhymes with `word` under the full rule."""
    rp = RHYMING_PART[word]
    return [
        w for w, other in RHYME_POOL.items()
        if other == rp and w != word and not shares_onset(word, w)
    ]


def family_size(word: str) -> int:
    """Rhyme-family size: pool words sharing this word's rime, excluding
    identical-onset pairs (see `shares_onset`). Computed from each word's
    primary pronunciation only (see module docstring)."""
    word = word.lower()
    if word not in PRIMARY:
        raise ValueError(f"{word!r} not found in CMUdict")
    if word not in RHYMING_PART:
        raise ValueError(f"{word!r} has no primary-stressed vowel in CMUdict")
    counts = _ONSET_COUNTS[RHYMING_PART[word]]
    total = sum(counts.values()) - (1 if word in RHYME_POOL else 0)
    onset = ONSET.get(word, "")
    if not onset:  # clause cannot fire on a vowel-initial word
        return total
    return total - (counts[onset] - (1 if word in RHYME_POOL else 0))


def syllable_count(word: str) -> int:
    """Vowel-phone count from the word's primary CMUdict pronunciation
    (phones ending in a stress digit 0/1/2 are vowels; consonants carry no
    stress digit)."""
    word = word.lower()
    if word not in PRIMARY:
        raise ValueError(f"{word!r} not found in CMUdict")
    return sum(1 for p in PRIMARY[word].split() if p[-1] in "012")


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


def difficulty_label(size: int) -> str:
    """Descriptive difficulty label: the integer part of log10(family_size).

    hard = 1-9, medium = 10-99, easy = 100+. Reported alongside the count, not
    used to select or to carry a claim - the analysis regresses on
    log10(family_size) directly. Coarse by design: "hard" spans a 9x range, so
    a word's actual count is the thing to quote. `article`, with exactly one
    rhyme in the entire language, is worth stating as its own fact rather than
    folding into a label.
    """
    if size < 1:
        return "none"
    return DIFFICULTY_LABELS.get(min(len(str(size)) - 1, 2), "easy")


def is_affixal(a: str, b: str) -> bool:
    """Screen for a morphological pair: one word is a prefix or suffix of the
    other (en+titled, title+d).

    Over-inclusive on purpose - it flags particle/article and adoption/option,
    where neither word derives from the other - so it is a prompt for human
    judgement, not a filter. The concern is a task confound rather than rhyme
    validity: if a word's only rhyme is a morphological superset of it, the
    model can complete the couplet by affixation instead of by phonological
    retrieval, plausibly exercising a different circuit from the one being
    measured.

    Concentrated at the low end of the axis, which is where sampling is
    tightest: of frequency-band words with exactly one rhyme, 39% have only an
    affixal partner, against 5.5% at 2-9 and 0.4% at 10-40.
    """
    lo, hi = sorted((a, b), key=len)
    return len(lo) < len(hi) and (hi.startswith(lo) or hi.endswith(lo))


MEMBERS_SHOWN = 3


def candidate_rows(band: set[str]) -> list[tuple[list[tuple[str, int]], str, float, bool]]:
    """One row per rhyme family, with the data the manual pass needs: up to
    `MEMBERS_SHOWN` usable words from the family and their rhyme counts, the
    family's most usable partner and that partner's frequency, and the affixal
    screen.

    Replaces the three per-band shortlists. Selection is now judgement against
    a range-spanning list, because the analysis is continuous in
    log10(family_size) and bands are only descriptive - so band machinery
    would constrain sampling without carrying any claim. It also removes an
    artifact of that machinery: no rule could produce a word between 21 and 53
    rhymes, since the medium cap sat at 20 and the easy band began at 54.

    Several words per family are shown, ranked by frequency, because which
    member of a family is usable is a sentence-writability judgement rather
    than a phonological one - "late" and "ate" are the same design point, but
    only one of them writes a natural line. An earlier version kept the
    alphabetically first member and dropped the rest, which hid that choice
    (and biased the list toward a- and b-words). Members of one family differ
    slightly in count, since the onset clause subtracts a different number of
    same-onset words for each, so each is listed with its own.
    """
    families: dict[str, list[tuple[str, float]]] = {}
    for word in band:
        if word not in RHYMING_PART:
            continue
        if nltk.pos_tag([word])[0][1] not in CONTENT_TAGS:
            continue
        families.setdefault(RHYMING_PART[word], []).append((word, zipf_frequency(word, "en")))

    rows: list[tuple[list[tuple[str, int]], str, float, bool]] = []
    for members in families.values():
        ranked = [w for w, _ in sorted(members, key=lambda m: (-m[1], m[0]))]
        # The onset clause can leave one member of a family with no partner
        # while others still have several, so head on the first that does.
        head = next((w for w in ranked if rhyme_partners(w)), None)
        if head is None:
            continue
        ranked = [head] + [w for w in ranked if w != head]
        partners = rhyme_partners(head)
        best = max(partners, key=lambda p: zipf_frequency(p, "en"))
        rows.append(
            (
                [(w, family_size(w)) for w in ranked[:MEMBERS_SHOWN]],
                best,
                zipf_frequency(best, "en"),
                all(is_affixal(head, p) for p in partners),
            )
        )
    return sorted(rows, key=lambda r: r[0][0][1])


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


def grid_coverage(entries: list[dict], targets: list[int] = GRID_TARGETS) -> list[dict]:
    """For each `GRID_TARGETS` value, the word in `entries` (read from
    `prompt_set.json`, not hardcoded) whose `rhymes_all` is closest in
    log10 space.

    Exists so the target-to-word mapping is derived from the current file,
    not hand-copied into a table that can silently go stale or contain a
    transcription error - a manual version of this table briefly conflated
    `reversed` (rhymes_all=36) with `become` (71) for the same target.

    Entries with `rhymes_all: null` (e.g. the "original" replication-anchor
    entry, a cross-word near-rhyme the metric can't score) are excluded from
    the search - they're exempt from the grid, not a candidate for it.
    """
    scoreable = [e for e in entries if e.get("rhymes_all") is not None]
    results = []
    for target in targets:
        best = min(
            scoreable, key=lambda e: abs(math.log10(e["rhymes_all"]) - math.log10(target))
        )
        gap = abs(math.log10(best["rhymes_all"]) - math.log10(target))
        results.append(
            {
                "target": target,
                "word": best["rhyme_word"],
                "rhymes_all": best["rhymes_all"],
                "log10_gap": round(gap, 3),
            }
        )
    return results


def print_distribution_summary(label: str, sizes: dict[str, int]) -> None:
    sorted_sizes = sorted(sizes.values())
    zero = sum(1 for n in sorted_sizes if n == 0)
    print(f"{label}: {len(sizes)} words, {zero} zero-rhyme ({100 * zero / len(sizes):.1f}%)")
    for p in (10, 25, 33, 50, 67, 75, 90):
        print(f"  {p}th percentile: {percentile(sorted_sizes, p):.1f}")


def load_tokenizers(models: list[str] = TOKENIZER_MODELS) -> dict:
    """Load one `AutoTokenizer` per model. Imports `transformers` locally so
    plain `python tools/prompt_set.py` (no --check-tokenizer) has zero HF/
    model dependency."""
    from transformers import AutoTokenizer

    return {m: AutoTokenizer.from_pretrained(m) for m in models}


def tokenizers_share_vocab(tokenizers: dict) -> bool:
    """Whether every tokenizer in `tokenizers` has an identical vocabulary.
    Verifies rather than assumes the three Gemma-3 sizes share one tokenizer
    - a prior pipeline stage only ever checked google/gemma-3-4b-pt."""
    vocabs = [t.get_vocab() for t in tokenizers.values()]
    return all(v == vocabs[0] for v in vocabs[1:])


def tokenize_leading_space(word: str, tokenizers: dict) -> dict[str, list[str]]:
    """Per-model token strings for `" word"` (leading space, mid-sentence
    form) via add_special_tokens=False."""
    form = " " + word
    result = {}
    for name, tok in tokenizers.items():
        ids = tok(form, add_special_tokens=False)["input_ids"]
        result[name] = [tok.decode([i]) for i in ids]
    return result


def build_tokenizer_report(
    existing_entries: list[dict],
    named_candidates: dict[str, list[str]],
    gap_options: list[dict],
    models: list[str] = TOKENIZER_MODELS,
) -> dict:
    tokenizers = load_tokenizers(models)
    shared = tokenizers_share_vocab(tokenizers)
    reference_model = models[0]

    entries = []

    for entry in existing_entries:
        word = entry["rhyme_word"]
        tokens_by_model = tokenize_leading_space(word, tokenizers)
        tokens = tokens_by_model[reference_model]
        entries.append(
            {
                "word": word,
                "source": "existing_prompt_set",
                "slug": entry["slug"],
                "rhymes_all": entry["rhymes_all"],
                "syllables": syllable_count(word),
                "leading_space_form": " " + word,
                "tokens": tokens,
                "tokens_by_model": tokens_by_model,
                "single_token": len(tokens) == 1,
            }
        )

    for gap, words in named_candidates.items():
        for word in words:
            tokens_by_model = tokenize_leading_space(word, tokenizers)
            tokens = tokens_by_model[reference_model]
            try:
                size = family_size(word)
            except ValueError:
                size = None
            entries.append(
                {
                    "word": word,
                    "source": "named_candidate",
                    "target_gap": gap,
                    "rhymes_all": size,
                    "syllables": syllable_count(word) if word in PRIMARY else None,
                    "leading_space_form": " " + word,
                    "tokens": tokens,
                    "tokens_by_model": tokens_by_model,
                    "single_token": len(tokens) == 1,
                }
            )

    for option in gap_options:
        word = option["word"]
        tokens_by_model = tokenize_leading_space(word, tokenizers)
        tokens = tokens_by_model[reference_model]
        entries.append(
            {
                "word": word,
                "source": "gap_fill_option",
                "target_gap": "6_8",
                "rhymes_all": option["rhymes_all"],
                "best_rhyme_partner": option["best_rhyme_partner"],
                "best_rhyme_partner_zipf": option["best_rhyme_partner_zipf"],
                "affixal_only_rhyme": option["affixal_only_rhyme"],
                "syllables": option["syllables"],
                "leading_space_form": " " + word,
                "tokens": tokens,
                "tokens_by_model": tokens_by_model,
                "single_token": len(tokens) == 1,
            }
        )

    existing_splits = [
        e["word"] for e in entries if e["source"] == "existing_prompt_set" and not e["single_token"]
    ]
    named_results = {
        e["word"]: e["single_token"] for e in entries if e["source"] == "named_candidate"
    }
    gap_6_8 = [e for e in entries if e["source"] == "gap_fill_option"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_models_checked": models,
        "tokenizer_shared_across_sizes": shared,
        "entries": entries,
        "summary": {
            "total_checked": len(entries),
            "existing_prompt_set_splits": existing_splits,
            "named_candidate_results": named_results,
            "gap_6_8_options": [e["word"] for e in gap_6_8],
        },
    }


def print_tokenizer_table(report: dict) -> None:
    print(f"\nShared tokenizer across {report['tokenizer_models_checked']}: "
          f"{report['tokenizer_shared_across_sizes']}")
    print(f"\n{'word':<14}{'source':<20}{'rhymes_all':<12}{'syllables':<11}{'single_token'}")
    for e in report["entries"]:
        rhymes_all = e["rhymes_all"] if e["rhymes_all"] is not None else "?"
        syllables = e["syllables"] if e["syllables"] is not None else "?"
        print(
            f"{e['word']:<14}{e['source']:<20}{str(rhymes_all):<12}"
            f"{str(syllables):<11}{e['single_token']}"
        )
    splits = report["summary"]["existing_prompt_set_splits"]
    print(f"\nExisting prompt_set.json words that split into multiple tokens: {splits or 'none'}")


def run_tokenizer_check() -> None:
    with open(PROMPT_SET_PATH, encoding="utf-8") as f:
        existing_entries = json.load(f)

    print("Checking tokenizer(s) against prompt_set.json...")
    report = build_tokenizer_report(existing_entries, {}, [])

    print_tokenizer_table(report)

    TOKENIZER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKENIZER_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {TOKENIZER_REPORT_PATH}")


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

    rows = candidate_rows(band)
    print(
        f"\nRange-spanning candidates: {len(rows)} rhyme families, "
        f"up to {MEMBERS_SHOWN} words each, "
        f"family_size {rows[0][0][0][1]}-{rows[-1][0][0][1]}"
    )
    by_difficulty: dict[str, int] = {}
    for members, _, _, _ in rows:
        label = difficulty_label(members[0][1])
        by_difficulty[label] = by_difficulty.get(label, 0) + 1
    print("  by difficulty: " + ", ".join(f"{k}={v}" for k, v in by_difficulty.items()))
    affixal = sum(1 for r in rows if r[3])
    print(f"  only-affixal rhyme (needs a human call): {affixal}")

    print("\n  sample across the range (family words, best partner, its zipf):")
    for target in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        near = [r for r in rows if r[0][0][1] == target]
        if near:
            members, p, z, aff = max(near, key=lambda r: r[2])
            words = ", ".join(f"{w}({n})" for w, n in members)
            print(
                f"    {target:>4}  {words:<40} -> {p:<14} zipf {z:.1f}"
                + ("  [affixal]" if aff else "")
            )

    print("\nRepetition-control candidates (content-word POS tags, abundant rhymes):")
    control = repetition_control_candidates(banded_sizes)
    print(", ".join(f"{w}({n},{tag})" for w, n, tag in control))

    # Sanity checks: "it" resolves via its primary pronunciation; "tear" and
    # "again" no longer inherit a rhyme family from their secondary
    # pronunciation (see module docstring); the onset clause must NOT fire on
    # article/particle, which is a load-bearing prompt-set pick.
    print(f"\nSanity check - 'it' family_size: {family_size('it')}")
    print(
        "Sanity check - onset clause on article/particle (must be False): "
        f"{shares_onset('particle', 'article')}"
    )
    print(
        "Sanity check - onset clause on theater/theatre (must be True):  "
        f"{shares_onset('theater', 'theatre')}"
    )
    tear_rhymes = [w for w in RHYME_POOL if w != "tear" and RHYME_POOL[w] == RHYMING_PART["tear"]]
    again_rhymes = [
        w for w in RHYME_POOL if w != "again" and RHYME_POOL[w] == RHYMING_PART["again"]
    ]
    print(f"Sanity check - 'tear' rhymes (primary pronunciation only): {tear_rhymes[:10]}")
    print(f"Sanity check - 'again' rhymes (primary pronunciation only): {again_rhymes[:10]}")

    print(f"\nGrid coverage ({GRID_TARGETS} targets vs current prompt_set.json):")
    with open(PROMPT_SET_PATH, encoding="utf-8") as f:
        prompt_set_entries = json.load(f)
    for row in grid_coverage(prompt_set_entries):
        print(
            f"    target {row['target']:>4} -> {row['word']:<12} "
            f"(rhymes_all={row['rhymes_all']}, log10 gap {row['log10_gap']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-tokenizer",
        action="store_true",
        help="Tokenizer-check prompt_set.json's words plus gap-fill candidates against "
        "the gated Gemma-3 tokenizers, writing experiment/tokenizer_check_report.json. "
        "Requires transformers and HF access to google/gemma-3-{270m,1b-pt,4b-pt}.",
    )
    args = parser.parse_args()
    if args.check_tokenizer:
        run_tokenizer_check()
    else:
        main()
