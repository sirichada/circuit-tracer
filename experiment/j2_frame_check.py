"""J2: does the chat prefix change what the model generates?

Background
----------
Generation ran on the bare prompt; attribution ran on `CHAT_PREFIX + prompt`,
four tokens apart. The graphs therefore describe a context the generator never
saw. The open question is whether that gap is
bookkeeping or a real divergence.

An earlier probe on this machine (`experiment/chat_prefix_probe_4b_output.log`)
found that `apply_chat_template` collapses rhyming entirely -- 0/11, every
continuation echoing the prompt's own final word. That is a *third* frame,
not ours: the template emits a closed user turn plus `<start_of_turn>model\n`,
which puts the model in answer-a-user mode. Our `CHAT_PREFIX` opens a user turn
and never closes it, leaving the model mid-utterance in continuation mode. So
that result does not settle J2 either way.

The single-step check (findings_5070.md 4a) does not settle it either, for the
reason that report gives: it conditions on the already-generated tokens, and
those tokens carry the mode. A frame difference that shows up as a change in
*what gets generated* is invisible to a measurement that fixes what was
generated.

This script runs the only test that separates them: free greedy generation from
scratch under both frames, same prompts, same stop rule, same labeller.

Reading the result
------------------
The verdict is the **rhyme rate under each frame**, not per-slug agreement:

* Same rate -> the frame does not change the behaviour. If both arms are at
  zero, that is uninformative rather than reassuring -- the size has no rhyming
  for the frame to affect.
* Different rate -> the graphs were attributed in a frame the model rhymes at a
  different rate in. Generation has to be redone in the attributed frame, which
  means new graphs, and the gap is a paper-level finding, not a bug.

Run this on a size that actually rhymes. 270M produces none in either frame, so
it cannot answer the question; 4B is where the earlier probe saw 6/11 vs 0/11.

The comparison against `rhyme_labels.json` is **reported, never a gate**. It
asks whether this machine reproduces continuations decoded on other hardware --
bf16 greedy argmax flips on near-ties, so a mismatch is expected across GPUs.
Both arms here run on one GPU with one set of kernels, so that drift hits them
equally and cannot manufacture a difference between them.

    python tests/j2_frame_check.py --size 4b            # the one worth running
    python tests/j2_frame_check.py --size 4b --loader replacement

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
# rather than being restated here -- three copies of the checkpoint names is how
# they went stale last time.
MODELS = {size: cfg.model_name for size, cfg in CONFIGS.items()}

WIDTH = "16k"  # matches generation-gemma-3-*.py
L0 = "small"

# Labels that count as the model having rhymed, for the per-arm rate.
RHYMED = {"rhyme", "near_rhyme"}
MAX_STEPS = 20  # matches generation-gemma-3-*.py
PROMPT_SET_PATH = REPO / "tools" / "prompt_set.json"
LABELS_PATH = REPO / "experiment" / "rhyme_labels.json"
OUT_PATH = REPO / "experiment" / "j2_frame_check_{size}.json"


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
    `tracing.py`'s `{(r["size"], r["slug"]): r}`. Keying on `slug` alone silently
    lets each size overwrite the previous one, so the last size in the file wins
    and every other size gets compared against the wrong labels. That happened:
    a 1B run was scored against 4B's rhymes and reported six unidirectional
    "hardware drift" mismatches (p = 0.031) that were pure artifact.
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

    # --- the verdict: per-arm rhyme rate ------------------------------------
    # Not per-slug label agreement. Greedy decoding at bf16 flips on near-ties,
    # and a single flip moves one slug between `none` and `near_rhyme` without
    # the model rhyming any more or less often. Agreement counts every such flip
    # as divergence; a rate does not. The 5070 run showed why this matters --
    # swapping only the forward-pass implementation moved agreement from 9/11 to
    # 6/11, i.e. the per-slug readout was reporting noise at the same magnitude
    # as any real effect. Rates are what the 4B probe made visible (6/11 vs
    # 0/11) and are what the paper would report.
    bare_rate = sum(r["bare"]["label"] in RHYMED for r in rows)
    pref_rate = sum(r["prefixed"]["label"] in RHYMED for r in rows)

    print(f"\n  bare      : {bare_rate}/{n} rhymed")
    print(f"  prefixed  : {pref_rate}/{n} rhymed")
    print(f"  per-slug labels agree on {n - len(diverged)}/{n} (detail, not the verdict)")

    # Paired design -- same 11 prompts under both frames -- so only the
    # discordant slugs carry information and the test is exact McNemar
    # (a two-sided sign test on them). A bare rate inequality is not a result:
    # at n=11 the 4B run gave 8/11 -> 6/11, which is three discordant slugs
    # splitting 2-1, i.e. p = 1.0. Reaching p < 0.05 here needs >= 6 flips all
    # in the same direction, which is exactly what the apply_chat_template probe
    # had (6/11 -> 0/11, p = 0.031) and what the literal CHAT_PREFIX does not.
    down = [r["slug"] for r in rows
            if r["bare"]["label"] in RHYMED and r["prefixed"]["label"] not in RHYMED]
    up = [r["slug"] for r in rows
          if r["prefixed"]["label"] in RHYMED and r["bare"]["label"] not in RHYMED]
    p_value = mcnemar_exact(len(down), len(up))

    print(f"  discordant: {len(down)} lost ({', '.join(down) or '-'}), "
          f"{len(up)} gained ({', '.join(up) or '-'})")
    print(f"  exact McNemar (two-sided): p = {p_value:.3f}")

    if not down and not up:
        print(f"\nJ2: no slug changed rhyme status under the frame ({bare_rate}/{n} either way).")
        if bare_rate == 0:
            print("Both arms are at zero, so this size has no rhyming behaviour for")
            print("the frame to change -- uninformative about J2 rather than evidence")
            print("for it. Ask a size that actually rhymes.")
    elif p_value < 0.05:
        print(f"\nJ2 OPEN: the frame changes the rhyme rate, {bare_rate}/{n} -> "
              f"{pref_rate}/{n} (p = {p_value:.3f}).")
        print("The graphs were attributed in a frame the model rhymes at a different")
        print("rate in. Regenerating them in the attributed frame is warranted, and")
        print("the gap is itself a finding about the attribution frame.")
    else:
        print(f"\nJ2: {bare_rate}/{n} -> {pref_rate}/{n}, not distinguishable from "
              f"decoding noise (p = {p_value:.3f}).")
        print("Not evidence that the frames agree -- n=11 cannot detect a small")
        print("effect. It does mean regenerating the graphs is unjustified on this")
        print("evidence. Report the numbers as a measured limitation instead, and")
        print("compare the flip count against the cross-GPU note below: if they are")
        print("the same size, the frame perturbs labels about as much as changing")
        print("the graphics card does.")

    # --- cross-GPU reproducibility: reported, never a gate -------------------
    # This compares against continuations decoded on *other* hardware, so it
    # answers "are the graphs on disk reproducible here", not "is this test
    # valid". Both arms ran on one GPU with one set of kernels, so drift hits
    # them identically and cannot manufacture a difference between them.
    if control_mismatch:
        print(f"\nnote: {len(control_mismatch)}/{n} slugs do not reproduce "
              "rhyme_labels.json on this machine:")
        for slug, was, now in control_mismatch:
            print(f"  {slug}: shipped={was} regenerated={now}")
        print("  Expected when the graphs were generated on different hardware --")
        print("  bf16 greedy argmax flips on near-ties. It does not affect the")
        print("  bare-vs-prefixed comparison above, which is within-machine.")
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
