#!/usr/bin/env python3
"""Writes the edge0-strata server config for a model folder, from the container's environment,
and tunes the engine for this machine.

It does what `edge0-infer serve` does before it starts edge0-strata (see edge0-strata/EDGE0.md,
"Launch contract with edge0-infer"): it picks the template in edge0-strata/edge0/models/ from
`model_name` in the folder's model.json, fills in the placeholders, and applies the model alias,
GPU, context length and extra engine arguments. The difference is that the result is started with
Strata's own server (serve/server.py) listening on the container's port, so its web page is served too.

Usage:
  edge0-strata-config.py config <output.json>   write the server config (with this machine's tuning, if saved)
  edge0-strata-config.py tune                   measure the tuning with Strata's tools/calibrate.py and save it

The tuning is saved in the model folder as strata-tuning.json, one entry per machine and model setup
(GPU, VRAM, CPU, RAM, model, model options, context), so a change to any of those needs a new tune.
Nothing outside the model folder is read or written.
"""
import json
import os
import subprocess
import sys
import time
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
TUNING_FILE = "strata-tuning.json"


def die(message):
    print(f"edge0: {message}", file=sys.stderr)
    sys.exit(1)


def env(name):
    return os.environ.get(name, "").strip()


def with_arg(args, flag, value):
    """`args` with `flag value` set (replaced if present), or removed when value is None (as tools/calibrate.py)."""
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def first_line(path, prefix):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def tuning_key(name, options, config):
    """What a tune is valid for: this GPU, CPU and RAM, and the model with its options and context."""
    try:
        gpu = subprocess.run(["nvidia-smi", "-i", str(config["gpu"]), "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        gpu_name, vram = [x.strip() for x in gpu.strip().split(",")]
        vram = f"{int(vram) / 1024:.0f}GB"
    except (OSError, ValueError, subprocess.SubprocessError):
        gpu_name, vram = "?", "?"
    ram = first_line("/proc/meminfo", "MemTotal").split()[0]
    ram = f"{int(ram) / 2**20:.0f}GB" if ram.isdigit() else "?"
    args = config["args"]
    context = args[args.index("--max-context") + 1] if "--max-context" in args else "?"
    return "|".join([gpu_name, vram, first_line("/proc/cpuinfo", "model name"), ram, name,
                     f"du={options['du']}", f"ple-io={options['ple-io']}", context])


def build_config(apply_tuning):
    """(config, tuning key, tuning file) for the model folder the environment names."""
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
    template_flags = [a for a in args if a.startswith("--")]
    if env("CONTEXT"):
        if not env("CONTEXT").isdigit():
            die(f"CONTEXT={env('CONTEXT')!r} is not a number")
        if "--max-context" in args:
            args[args.index("--max-context") + 1] = env("CONTEXT")
        else:
            args += ["--max-context", env("CONTEXT")]
    key = tuning_key(name, options, config)
    tuning_file = model_dir / TUNING_FILE
    if apply_tuning:
        tuned = None
        if env("TUNING").lower() == "off":
            print("edge0: TUNING=off: the engine runs with its default settings", file=sys.stderr)
        elif tuning_file.exists():
            try:
                tuned = json.loads(tuning_file.read_text()).get("entries", {}).get(key)
            except (OSError, ValueError, AttributeError) as exc:
                print(f"edge0: warning: ignoring {tuning_file}: {exc}", file=sys.stderr)
            if tuned is None:
                print(f"edge0: {TUNING_FILE} has no tune for this setup ({key}); run the tune command again",
                      file=sys.stderr)
        if tuned:
            for flag, value in (tuned.get("settings") or {}).items():
                args = with_arg(args, flag, value)
            config.setdefault("env", {}).update(tuned.get("env") or {})
            chosen = " ".join(f"{k} {v}" for k, v in (tuned.get("settings") or {}).items()) or "the defaults"
            print(f"edge0: tuned for this machine on {tuned.get('date', '?')}: {chosen}", file=sys.stderr)

    # STRATA_ARGS: machine tuning given by hand. It may override a tuned flag, not the template's own.
    words = env("STRATA_ARGS").split()
    for word in words:
        if word == "--max-context":
            die("STRATA_ARGS: use CONTEXT instead of --max-context")
        if word.startswith("--") and word in template_flags:
            die(f"STRATA_ARGS: {word} is set by the model template")
        if word.startswith("--") and word in args:
            args = with_arg(args, word, None)
    args += words
    config["args"] = args

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

    print(f"edge0: model {name} from {model_dir} as {config['model_name']!r} on GPU {config['gpu']}", file=sys.stderr)
    return config, key, tuning_file


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def tune():
    """Measure with Strata's calibration (tools/calibrate.py) and save the result for this setup."""
    config, key, tuning_file = build_config(apply_tuning=False)
    if not os.access(tuning_file.parent, os.W_OK):
        die(f"{tuning_file.parent} is read-only: the tune is saved there as {TUNING_FILE}")
    sys.path[:0] = [str(STRATA_DIR / "tools"), str(STRATA_DIR)]
    os.chdir(STRATA_DIR)
    import calibrate
    print(f"edge0: tuning for {key}", file=sys.stderr)
    print("edge0: this takes roughly 15-30 minutes and restarts the engine several times", file=sys.stderr)
    started = time.time()
    res = calibrate.run(config, say=lambda *a: print(*a, flush=True))
    data = {"version": 1, "entries": {}}
    if tuning_file.exists():
        try:
            data = json.loads(tuning_file.read_text())
            data.setdefault("entries", {})
        except ValueError:
            print(f"edge0: warning: replacing unreadable {tuning_file}", file=sys.stderr)
            data = {"version": 1, "entries": {}}
    entry = {"settings": res.get("settings") or {}, "env": res.get("env") or {},
             "tok_s": (res.get("report") or {}).get("tok_s"), "date": time.strftime("%Y-%m-%d"),
             "minutes": round((time.time() - started) / 60, 1), "report": res.get("report") or {}}
    data["entries"][key] = entry
    write_json(tuning_file, data)
    chosen = " ".join(f"{k} {v}" for k, v in entry["settings"].items()) or "the defaults are already the fastest here"
    print(f"edge0: saved to {tuning_file}: {chosen}" + (f" ({entry['tok_s']} tok/s)" if entry["tok_s"] else ""))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "config":
        config = build_config(apply_tuning=True)[0]
        out = Path(sys.argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, config)
    elif len(sys.argv) == 2 and sys.argv[1] == "tune":
        tune()
    else:
        die("usage: edge0-strata-config.py config <output.json> | tune")


if __name__ == "__main__":
    main()
