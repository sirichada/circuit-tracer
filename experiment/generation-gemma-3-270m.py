import os
import re
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set
from circuit_tracer.utils import create_graph_files

MODEL_NAME = "google/gemma-3-270m-it"
TRANSCODER_REPO = "google/gemma-scope-2-270m-it"
NUM_LAYERS = 18
OUTPUT_DIR = "./graphs/gemma-3-270m-it"
PROMPT_SET_PATH = Path(__file__).parent.parent / "tools" / "prompt_set.json"

WIDTH = "16k"
L0 = "small"

max_n_logits = 5
desired_logit_prob = 0.95
max_feature_nodes = 1028
batch_size = 8
offload = "disk"
verbose = True

MAX_STEPS = 20

# attribute() requires -it inputs to start with this exact 4-token prefix
# (see replacement_model_transformerlens.py:389-399); those positions are
# zeroed out of the transcoder activations before the graph is built, so
# this has no effect on the graph itself. Generation is unaffected — it
# never sees this prefix, only attribute() does.
CHAT_PREFIX = "<bos><start_of_turn>user\n"


def load_prompt_set() -> list[dict]:
    return json.loads(PROMPT_SET_PATH.read_text())


def generate_and_attribute(model, tokenizer, prompt_text: str, slug: str) -> None:
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    STOP_IDS = {107, 108, tokenizer.eos_token_id}
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0:
        STOP_IDS.add(end_of_turn_id)
    token_trace = []

    for step in range(MAX_STEPS):
        with torch.no_grad():
            out = model(input_ids)

        logits_last = out[0, -1, :].float()
        probs = F.softmax(logits_last, dim=-1)
        top_probs, top_ids = torch.topk(probs, 10)
        next_id = top_ids[0].item()

        token_trace.append({
            "step": step,
            "token_id": next_id,
            "token_str": tokenizer.decode([next_id]),
            "chosen_prob": top_probs[0].item(),
            "chosen_logit": logits_last[next_id].item(),
            "top10": [{"token": tokenizer.decode([tid.item()]), "prob": p.item()}
                      for tid, p in zip(top_ids, top_probs)],
        })

        if next_id in STOP_IDS:
            break

        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    print(f"[{slug}] Generated:",
          "".join(t["token_str"] for t in token_trace if t["token_id"] not in STOP_IDS))

    for i, token in enumerate(token_trace):
        if token["token_id"] in STOP_IDS:
            break

        tokens_so_far = "".join(t["token_str"] for t in token_trace[:i])
        prompt_at_step = prompt_text + tokens_so_far

        graph = attribute(
            prompt=CHAT_PREFIX + prompt_at_step,
            model=model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            max_feature_nodes=max_feature_nodes,
            batch_size=batch_size,
            offload=offload,
            verbose=verbose,
        )

        safe_token_str = re.sub(r'[\\/:*?"<>|]', "_", token["token_str"].strip())
        step_slug = f"step-{i:02d}-{safe_token_str.replace(' ', '_')}"
        create_graph_files(
            graph_or_path=graph,
            slug=step_slug,
            output_path=f"{OUTPUT_DIR}/{slug}",
            node_threshold=0.8,
            edge_threshold=0.98,
        )
        print(f"[{slug}] ✓ step {i:02d} → '{token['token_str']}'  saved as '{step_slug}'")

        del graph
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    transcoder_paths = {}
    for layer in range(NUM_LAYERS):
        path = hf_hub_download(
            repo_id=TRANSCODER_REPO,
            filename=f"transcoder_all/layer_{layer}_width_{WIDTH}_l0_{L0}/params.safetensors",
        )
        transcoder_paths[layer] = path

    transcoder_set = load_transcoder_set(
        transcoder_paths=transcoder_paths,
        scan=TRANSCODER_REPO.split("/")[-1],
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lazy_encoder=False,
        lazy_decoder=True,
        special_load_fn="gemma-scope-2",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=MODEL_NAME,
        transcoders=transcoder_set,
        backend="transformerlens",
        dtype=torch.bfloat16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    prompt_set = load_prompt_set()

    for record in prompt_set:
        slug = record["slug"]
        prompt_text = record["prompt_text"]

        slug_dir = Path(OUTPUT_DIR) / slug
        if slug_dir.exists() and any(slug_dir.iterdir()):
            print(f"[{slug}] already has output — skipping")
            continue
        
        try:
            generate_and_attribute(model, tokenizer, prompt_text, slug)
        except Exception:
            print(f"[{slug}] FAILED — skipping, see traceback below")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
