"""Does the chat prefix change what the model generates?

Standalone diagnostic, not part of the main five-stage pipeline. Runs free
greedy generation from scratch under both the bare prompt and
`CHAT_PREFIX + prompt`, same prompts/stop rule/labeller, and reports the
rhyme rate under each frame plus an exact McNemar test over the paired result.

    python experiment/chat_prefix_frame_check.py --size 4b            # the one worth running
    python experiment/chat_prefix_frame_check.py --size 4b --loader replacement

No transcoders and no attribution: this is base-model greedy decoding only.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "experiment"))

from tracing import CONFIGS  # noqa: E402

from rhyme_labels import (  # noqa: E402
    CHAT_PREFIX,
    classify,
    inflected_repeat,
    is_looped,
    last_content_word,
    line_overlap,
    normalise_line,
    prompt_line,
)

# Model names, transcoder repos and layer counts come from tracing.py's CONFIGS
# rather than being restated here, so there is only one place to keep them current.
MODELS = {size: cfg.model_name for size, cfg in CONFIGS.items()}

WIDTH = "16k"  # matches generation-gemma-3-*.py
L0 = "small"

# Labels that count as the model having rhymed, for the per-arm rate.
RHYMED = {"rhyme", "near_rhyme"}
MAX_STEPS = 20  # matches generation-gemma-3-*.py
PROMPT_SET_PATH = REPO / "tools" / "prompt_set.json"
LABELS_PATH = REPO / "experiment" / "rhyme_labels.json"
OUT_PATH = REPO / "results" / "j2_frame_check_{size}.json"


def load_model(size: str, loader: str):
    """Return (callable logits fn, tokenizer).

    `hf` is the light path and is the default. `replacement` loads the same
    class the generation scripts used, for anyone who wants arm A to match
    bit-for-bit rather than merely label-for-label.
    """
    name = MODELS[size]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)

    if loader == "replacement":
        # Mirrors generation-gemma-3-*.py:128-154 exactly. `from_pretrained` does
        # not take a transcoder-name string; the transcoders have to be built
        # first and handed to `from_pretrained_and_transcoders`.
        from huggingface_hub import hf_hub_download

        from circuit_tracer import ReplacementModel
        from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

        cfg = CONFIGS[size]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = {
            layer: hf_hub_download(
                repo_id=cfg.transcoder_repo,
                filename=f"transcoder_all/layer_{layer}_width_{WIDTH}_l0_{L0}/params.safetensors",
            )
            for layer in range(cfg.n_layers)
        }
        transcoders = load_transcoder_set(
            transcoder_paths=paths,
            scan=cfg.transcoder_repo.split("/")[-1],
            feature_input_hook="hook_resid_mid",
            feature_output_hook="hook_mlp_out",
            device=device,
            lazy_encoder=False,
            lazy_decoder=True,
            special_load_fn="gemma-scope-2",
        )
        model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=name,
            transcoders=transcoders,
            backend="transformerlens",
            dtype=torch.bfloat16,
            device=device,
        )

        def logits_of(input_ids):
            with torch.no_grad():
                return model(input_ids)[0, -1, :].float()
    else:
        from transformers import AutoModelForCausalLM

        # bf16 to match generation-gemma-3-*.py, not tracing.py. This script
        # has to reproduce what the model *wrote*, so it must use the dtype the
        # continuations were written at; tracing.py's fp32 is for resolving
        # small logit differences, a different job.
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        def logits_of(input_ids):
            with torch.no_grad():
                return model(input_ids.to(model.device)).logits[0, -1, :].float()

    return logits_of, tokenizer


def stop_ids(tokenizer) -> set[int]:
    """Same stop set the generation scripts used."""
    ids = {107, 108, tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        ids.add(eot)
    return {i for i in ids if isinstance(i, int) and i >= 0}


def generate(logits_of, tokenizer, text: str, add_special_tokens: bool) -> dict:
    """Greedy decode from `text` until a stop token or MAX_STEPS.

    `add_special_tokens` is the whole asymmetry between the two arms and is
    passed explicitly rather than defaulted: the bare prompt needs the
    tokenizer to prepend BOS, while CHAT_PREFIX already opens with a literal
    `<bos>` and would otherwise get a second one -- which is exactly the
    off-by-one this pipeline already got bitten by once.
    """
    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=add_special_tokens
    )["input_ids"]
    n_prompt_tokens = input_ids.shape[1]
    stops = stop_ids(tokenizer)
    pieces, stopped = [], None

    for _ in range(MAX_STEPS):
        next_id = int(torch.argmax(logits_of(input_ids)).item())
        if next_id in stops:
            stopped = next_id
            break
        pieces.append(tokenizer.decode([next_id]))
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    return {
        "text": "".join(pieces),
        "n_generated": len(pieces),
        "n_prompt_tokens": n_prompt_tokens,
        "stopped_on": stopped,
        "hit_max_steps": stopped is None,
    }


def mcnemar_exact(n_down: int, n_up: int) -> float:
    """Two-sided exact McNemar: a sign test over the discordant pairs.

    Concordant slugs (rhymed in both frames, or neither) carry no information
    about the frame and are correctly ignored -- which is why an unpaired
    comparison of 8/11 vs 6/11 overstates the evidence.
    """
    n = n_down + n_up
    if n == 0:
        return 1.0
    k = min(n_down, n_up)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(2 * tail, 1.0)


def label(generated: str, prompt_text: str, target: str) -> dict:
    """Mirror of rhyme_labels.label_one, minus the graph plumbing."""
    words = normalise_line(generated)
    lcw = last_content_word(generated)
    if lcw is None:
        return {"label": "degenerate", "rhyme_word": None, "looped": is_looped(words)}
    word, _ = lcw
    return {
        "label": classify(word, target),
        "rhyme_word": word,
        "looped": is_looped(words),
        "inflected_repeat": inflected_repeat(word, target),
        "line_overlap": line_overlap(words, prompt_line(prompt_text)),
    }


def shipped_labels(size: str) -> dict[str, dict]:
    """Labels for THIS size only, for the arm-A cross-check.

    `rhyme_labels.json` is a flat list of records keyed `(size, slug)` -- see
    `tracing.py`'s `{(r["size"], r["slug"]): r}`. Keying on `slug` alone would
    silently let each size overwrite the previous one, so the last size in the
    file wins and every other size gets compared against the wrong labels.
    """
    if not LABELS_PATH.exists():
        return {}
    rows = json.loads(LABELS_PATH.read_text())
    if not isinstance(rows, list):
        raise ValueError(
            f"{LABELS_PATH} is a {type(rows).__name__}, expected the flat list of "
            "(size, slug) records that rhyme_labels.py writes."
        )
    out = {r["slug"]: r for r in rows if r.get("size") == size}
    if not out:
        sizes = sorted({r.get("size") for r in rows})
        print(f"warning: no rhyme_labels.json rows for size {size!r} (file has {sizes})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", default="270m", choices=sorted(MODELS))
    ap.add_argument("--loader", default="hf", choices=("hf", "replacement"))
    ap.add_argument("--slugs", nargs="*", help="default: all 11")
    args = ap.parse_args()

    prompts = json.loads(PROMPT_SET_PATH.read_text())
    if args.slugs:
        prompts = [p for p in prompts if p["slug"] in args.slugs]

    logits_of, tokenizer = load_model(args.size, args.loader)
    shipped = shipped_labels(args.size)

    rows, diverged, control_mismatch = [], [], []
    for p in prompts:
        slug, text, target = p["slug"], p["prompt_text"], p["rhyme_word"]

        bare = generate(logits_of, tokenizer, text, add_special_tokens=True)
        pref = generate(logits_of, tokenizer, CHAT_PREFIX + text, add_special_tokens=False)

        bare_lab = label(bare["text"], text, target)
        pref_lab = label(pref["text"], text, target)
        agree = bare_lab["label"] == pref_lab["label"]
        if not agree:
            diverged.append(slug)

        # Control: does arm A reproduce what is already on disk?
        was = shipped.get(slug, {}).get("label")
        if was is not None and was != bare_lab["label"]:
            control_mismatch.append((slug, was, bare_lab["label"]))

        rows.append({
            "slug": slug,
            "target": target,
            "bare": {**bare, **bare_lab},
            "prefixed": {**pref, **pref_lab},
            "labels_agree": agree,
            "shipped_label": was,
        })

        flag = "  " if agree else "!!"
        print(f"{flag} {slug:10s} bare={bare_lab['label']:<12s} "
              f"prefixed={pref_lab['label']:<12s} "
              f"({bare_lab['rhyme_word']} / {pref_lab['rhyme_word']})")

    n = len(rows)

    # The verdict is the rate, not per-slug agreement.
    bare_rate = sum(r["bare"]["label"] in RHYMED for r in rows)
    pref_rate = sum(r["prefixed"]["label"] in RHYMED for r in rows)

    print(f"\n  bare      : {bare_rate}/{n} rhymed")
    print(f"  prefixed  : {pref_rate}/{n} rhymed")
    print(f"  per-slug labels agree on {n - len(diverged)}/{n} (detail, not the verdict)")

    down = [r["slug"] for r in rows
            if r["bare"]["label"] in RHYMED and r["prefixed"]["label"] not in RHYMED]
    up = [r["slug"] for r in rows
          if r["prefixed"]["label"] in RHYMED and r["bare"]["label"] not in RHYMED]
    p_value = mcnemar_exact(len(down), len(up))

    print(f"  discordant: {len(down)} lost ({', '.join(down) or '-'}), "
          f"{len(up)} gained ({', '.join(up) or '-'})")
    print(f"  exact McNemar (two-sided): p = {p_value:.3f}")

    if not down and not up:
        print(f"\nno slug changed rhyme status under the frame ({bare_rate}/{n} either way).")
    elif p_value < 0.05:
        print(f"\nframe changes the rhyme rate: {bare_rate}/{n} -> "
              f"{pref_rate}/{n} (p = {p_value:.3f}).")
    else:
        print(f"\n{bare_rate}/{n} -> {pref_rate}/{n}, not distinguishable from "
              f"decoding noise (p = {p_value:.3f}).")

    # Cross-GPU reproducibility is reported, never a gate.
    if control_mismatch:
        print(f"\nnote: {len(control_mismatch)}/{n} slugs do not reproduce "
              "rhyme_labels.json on this machine:")
        for slug, was, now in control_mismatch:
            print(f"  {slug}: shipped={was} regenerated={now}")
    elif shipped:
        print("\nnote: arm A reproduces rhyme_labels.json exactly.")
    else:
        print("\nnote: no rhyme_labels.json on disk, cross-check skipped.")

    out = Path(str(OUT_PATH).format(size=args.size))
    out.write_text(json.dumps({
        "size": args.size,
        "loader": args.loader,
        "model": MODELS[args.size],
        "max_steps": MAX_STEPS,
        "chat_prefix": CHAT_PREFIX,
        "n_prompts": n,
        "bare_rhyme_rate": bare_rate,
        "prefixed_rhyme_rate": pref_rate,
        "discordant_lost": down,
        "discordant_gained": up,
        "mcnemar_p": p_value,
        "n_diverged": len(diverged),
        "diverged": diverged,
        "control_mismatch": control_mismatch,
        "rows": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
