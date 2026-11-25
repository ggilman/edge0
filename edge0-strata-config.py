#!/usr/bin/env python3
"""Writes the edge0-strata server config for a model folder, from the container's environment.

It does what `edge0-infer serve` does before it starts edge0-strata (see edge0-strata/EDGE0.md,
"Launch contract with edge0-infer"): it picks the template in edge0-strata/edge0/models/ from
`model_name` in the folder's model.json, fills in the placeholders, and applies the model alias,
GPU, context length and extra engine arguments. The difference is that the result is started with
Strata's own server (serve/server.py) listening on the container's port, so its web page is served too.

Usage: edge0-strata-config.py <output.json>
"""
import json
import os
import sys
from pathlib import Path

STRATA_DIR = Path(os.environ.get("EDGE0_STRATA_DIR", "/opt/edge0/edge0-strata"))

# model.json "model_name" -> (template, default model name, expert format). As in edge0-infer's registry.
MODELS = {
    "edge0-145b-q3k": ("edge0-145b.json", "edge0_145b", "q3k"),
    "edge0-145b-fp4": ("edge0-145b.json", "edge0_145b", "fp4"),
    "edge0-120b-q3k": ("edge0-120b.json", "edge0_120b", "q3k"),
    "edge0-120b-fp4": ("edge0-120b.json", "edge0_120b", "fp4"),
}
MODEL_OPTIONS = {"du": ("q8", "q4"), "ple-io": ("direct", "ram")}
EFFORTS = ("none", "low", "medium", "high")


def die(message):
    print(f"edge0: {message}", file=sys.stderr)
    sys.exit(1)


def env(name):
    return os.environ.get(name, "").strip()


def main():
    if len(sys.argv) != 2:
        die("usage: edge0-strata-config.py <output.json>")
    out = Path(sys.argv[1])

    model_dir = Path(env("MODEL_DIR") or f"/models/{env('EDGE0_MODEL')}")
    try:
        name = json.loads((model_dir / "model.json").read_text())["model_name"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        die(f"cannot read model_name from {model_dir}/model.json: {exc}")
    if name not in MODELS:
        die(f"{model_dir} holds model {name!r}; supported: {', '.join(MODELS)}")
    template_name, default_alias, experts = MODELS[name]

    options = {"du": "q8", "ple-io": "direct"}
    for item in env("MODEL_OPTIONS").split():
        key, _, value = item.partition("=")
        if key not in MODEL_OPTIONS or value not in MODEL_OPTIONS[key]:
            die(f"MODEL_OPTIONS: unknown option {item!r}; use du=q8|q4 and ple-io=direct|ram")
        options[key] = value
    du_file = model_dir / f"latent_dense_{options['du']}.bin"
    for need in (du_file, model_dir / "experts.bin", model_dir / "model-skeleton.gguf", model_dir / "tokenizer"):
        if not need.exists():
            die(f"{model_dir} is incomplete: missing {need.name}")

    values = {
        "STRATA_DIR": str(STRATA_DIR),
        "EXE_SUFFIX": "",
        "MODEL_DIR": str(model_dir),
        "LAT_FUSED_Q3K": "1" if experts == "q3k" else "0",
        "DU_FILE": str(du_file),
        "DU_FMT": "4" if options["du"] == "q4" else "8",
        "PLE_IO": options["ple-io"],
    }

    def expand(value):
        if isinstance(value, str):
            for key, text in values.items():
                value = value.replace("${" + key + "}", text)
            if "${" in value:
                die(f"template {template_name} uses an unknown placeholder: {value}")
            return value
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    template = json.loads((STRATA_DIR / "edge0" / "models" / template_name).read_text())
    template["lib_dirs"] = [d for d in template.get("lib_dirs", []) if d != "${CUDA_LIB_DIR}"]
    template.pop("prepare", None)   # only the Metal templates have one
    config = expand(template)

    config["model_name"] = env("MODEL_ALIAS") or default_alias

    device = env("DEVICE") or "cuda:0"
    if device == "cuda":
        device = "cuda:0"
    if not device.startswith("cuda:") or not device[5:].isdigit():
        die(f"DEVICE={device!r}: use cuda or cuda:<n>")
    config["gpu"] = int(device[5:])

    args = config["args"]
    if env("CONTEXT"):
        if not env("CONTEXT").isdigit():
            die(f"CONTEXT={env('CONTEXT')!r} is not a number")
        if "--max-context" in args:
            args[args.index("--max-context") + 1] = env("CONTEXT")
        else:
            args += ["--max-context", env("CONTEXT")]
    template_args = list(args)
    for word in env("STRATA_ARGS").split():
        if word == "--max-context":
            die("STRATA_ARGS: use CONTEXT instead of --max-context")
        if word.startswith("--") and word in template_args:
            die(f"STRATA_ARGS: {word} is set by the model template")
        args.append(word)

    effort = env("REASONING_EFFORT").lower()
    thinking = env("THINKING").lower()
    if not effort and thinking == "off":
        effort = "none"
    elif thinking not in ("", "on", "off"):
        die(f"THINKING={thinking!r}: use on or off")
    if effort:
        if effort not in EFFORTS:
            die(f"REASONING_EFFORT={effort!r}: use one of {', '.join(EFFORTS)}")
        config["reasoning_effort"] = effort

    for key, var in (("trusted_origins", "TRUSTED_ORIGINS"), ("cors_origins", "CORS_ORIGINS")):
        items = [x for x in env(var).replace(",", " ").split() if x]
        if items:
            config[key] = items

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"edge0: model {name} from {model_dir} as {config['model_name']!r} on GPU {config['gpu']}", file=sys.stderr)


if __name__ == "__main__":
    main()
