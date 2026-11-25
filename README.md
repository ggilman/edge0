# edge0-infer for Docker (NVIDIA CUDA)

A container image and Compose file for running **Edge0 145B** and **Edge0 120B** as a local,
OpenAI- and Anthropic-compatible API server on a single NVIDIA GPU.

The image builds two programs from the upstream [Edge0](https://github.com/Edge0-AI/Edge0)
repository:

- **edge0-strata**: the CUDA engine (a fork of Strata) for Edge0 145B and 120B, and Strata's own
  server, which the container runs. It serves the API (`/v1/chat/completions`, `/v1/messages`,
  `/v1/responses`, `/v1/models`, `/health`) and Strata's **web page** (Chat, Images, Monitor, About)
  on the same port. These are mixture-of-experts models whose experts are spread across GPU
  memory, system RAM and disk, so a 145B model can run on one consumer GPU.
- **edge0-infer**: Edge0's CLI. It's in the image for its tools (`edge0-infer inspect`, `chat`,
  `bench`), but it doesn't serve: `edge0-infer serve` forwards only the API and has no web page.
  The container builds the engine config from the same templates `edge0-infer serve` uses
  (`edge0-strata/edge0/models/`), so the engine runs with the same settings.

This repository only packages that software. It is not part of the Edge0 project and isn't
affiliated with it. Model weights are not included; you download them separately.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability 8.0 or newer. The image is compiled for **sm_120 (RTX 50-series)** by default; see [Building for other GPUs](#building-for-other-gpus). One GPU per server. |
| Memory | The Edge0 docs give **16 GiB GPU memory + 20 GiB system RAM** as the minimum. More of either makes it faster. |
| Driver | An NVIDIA driver that supports **CUDA 13** (580 or newer). |
| Host software | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and Docker Compose v2. |
| Disk | About 50-110 GB per model folder, plus about 5 GB for the image. Use fast SSD storage: experts that don't fit in memory are read from disk. |

## Quick start

**1. Download a model folder.** Each folder holds one model in one expert format:

| Folder | Weights | Notes |
|---|---|---|
| [`Edge0-145B-q3k`](https://huggingface.co/Edge0/Edge0-145B-q3k) | 40.81 GiB | Q3_K experts. The only 145B build with a published result on a 16 GiB VRAM / 20 GiB RAM budget. **The default here.** |
| [`Edge0-145B-NVFP4`](https://huggingface.co/Edge0/Edge0-145B-NVFP4) | 47.79 GiB | NVFP4 experts. Needs more memory; no 16 GiB result is published. |
| [`Edge0-120B-q3k`](https://huggingface.co/Edge0/Edge0-120B-q3k) | 27.98 GiB | Smaller model, Q3_K. |
| [`Edge0-120B-NVFP4`](https://huggingface.co/Edge0/Edge0-120B-NVFP4) | 32.96 GiB | Smaller model, NVFP4. |

```sh
pip install -U huggingface_hub
hf download Edge0/Edge0-145B-q3k --local-dir ./models/Edge0-145B-q3k
```

Use the folder as provided: don't move files out of it or rename them. Check the model card on
Hugging Face for the weights' license.

**2. Configure.**

```sh
cp .env.example .env
# set API_KEY (for example: openssl rand -hex 24), and EDGE0_MODEL if you downloaded another folder
```

**3. Start it.**

```sh
docker compose up -d --build     # the first build compiles CUDA code and takes a while
docker compose logs -f           # wait for:  Listening: http://0.0.0.0:8080 model=edge0_145b
```

Loading takes about one to a few minutes, longer on the first start or from slow disks. The
server doesn't answer until loading is done; the container shows as `healthy` once it is serving.

**4. Try it.** Open `http://<host>:8080` in a browser for the web page. It asks for the API key
once and keeps it in that browser. Or use the API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"edge0_145b","messages":[{"role":"user","content":"Say hello."}]}'
```

Clients:

| | |
|---|---|
| OpenAI-compatible base URL | `http://<host>:8080/v1` |
| Anthropic-compatible base URL | `http://<host>:8080` (no `/v1`; e.g. Claude Code's `ANTHROPIC_BASE_URL`) |
| Auth | `Authorization: Bearer <API_KEY>` or `x-api-key: <API_KEY>` |
| Model name | `edge0_145b` or `edge0_120b`, or whatever you set with `MODEL_ALIAS` |

## Compose settings

Set these in `.env`, in your shell, or as stack environment variables in Portainer. Empty means
"use the default".

### Required

| Variable | Description |
|---|---|
| `API_KEY` | Key every API client must send, and the key you enter in the web page. Requests without it get `401`. The container won't start without one. |

### Model and storage

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `./models` | Host folder containing the model folders. Mounted at `/models`. **Must be writable**: the first start may write files (such as `mtp-q2_0.gguf`) into the model folder. |
| `EDGE0_MODEL` | `Edge0-145B-q3k` | Name of the model folder inside `MODELS_DIR` to serve. Switching models is a restart: change it and run `docker compose up -d`. |

### Network

| Variable | Default | Description |
|---|---|---|
| `EDGE0_PORT` | `8080` | Port on the host. |
| `EDGE0_BIND` | `0.0.0.0` | Host address to listen on. Use `127.0.0.1` to allow only local clients, e.g. behind a reverse proxy on the same machine. |
| `TRUSTED_ORIGINS` | | Space- or comma-separated addresses the web page is opened from when that differs from the server's own, e.g. behind a reverse proxy or tunnel: `https://strata.example.com`. |
| `CORS_ORIGINS` | | Space- or comma-separated origins of *other* web pages allowed to call `/v1/*` from a browser. |

### Server

| Variable | Default | Description |
|---|---|---|
| `CONTEXT` | `131072` | Maximum context length in tokens. Lower it to save memory. |
| `DEVICE` | `cuda:0` | GPU to run on, `cuda` or `cuda:<n>`. One GPU only. To pick the GPU from Docker's side instead, set `GPU_COUNT=1`. |
| `GPU_COUNT` | `all` | How many GPUs Docker gives the container. |
| `MODEL_ALIAS` | `edge0_145b` / `edge0_120b` | Model name the API reports (`/v1/models`). |
| `REASONING_EFFORT` | model default (`high`) | Thinking level for requests that name none: `none`, `low`, `medium` or `high`. The web page's Chat settings and a request's own level win. |
| `THINKING` | | `off` is the same as `REASONING_EFFORT=none`. Kept for older setups. |
| `MODEL_OPTIONS` | | Space-separated `name=value` model options. See [Model options](#model-options). |
| `STRATA_ARGS` | | Space-separated engine tuning flags, appended to the engine's command line. See [Engine tuning](#engine-tuning). |
| `TUNING` | | `off` ignores a saved tune (`strata-tuning.json`) and runs the engine's defaults. See [Tuning for this machine](#tuning-for-this-machine). |
| `SERVE_ARGS` | | Extra flags for Strata's `serve/server.py`, appended as-is, e.g. `--parallel 2` or `--api-monitor`. |
| `TZ` | `UTC` | Time zone for log timestamps. |

### Image and build

| Variable | Default | Description |
|---|---|---|
| `EDGE0_IMAGE` | `edge0-infer:cuda13` | Image to run or build, e.g. `yourname/edge0-infer:cuda13` once published. |
| `CUDA_ARCHITECTURES` | `120` | GPUs to compile for (build only). See below. |
| `EDGE0_REF` | `main` | Upstream Edge0 branch, tag or commit to build (build only). |
| `CONTAINER_NAME` | `edge0` | Container name. |

The Compose file also sets `memlock` to unlimited (required for `ple-io=ram`), `shm_size: 2gb`,
a 2-minute stop grace period so the engine can unload cleanly, and log rotation (3 × 20 MB).

### Model options

Passed as `MODEL_OPTIONS`, e.g. `MODEL_OPTIONS=du=q4 ple-io=ram`.

| Option | Values | Description |
|---|---|---|
| `du` | `q8` (default), `q4` | Precision of the dense up-projection. `q4` uses less memory. |
| `ple-io` | `direct` (default), `ram` | `ram` keeps the per-layer embeddings in system RAM instead of reading them from disk. Needs the unlimited `memlock` this Compose file sets. |

### Tuning for this machine

Some engine settings depend on the computer more than on the model: the share of missing experts
copied over PCIe (`--pcie-frac`), the draft layer's confidence floor (`--spec-min-p`), the CPU
thread count (`--pool-workers`, which matters on CPUs with efficiency cores) and how the VRAM expert
cache follows the conversation (`--adapt-*`). The `tune` command measures them with Strata's own
tuner (`tools/calibrate.py`, what Strata's `--calibrate` runs) and keeps a value only when it is more
than 3% faster than the default.

It loads the model itself, so **stop the server first** (two copies don't fit in memory), and run it
with the same settings as the server:

```sh
docker compose stop edge0
docker compose run --rm edge0 tune     # roughly 15-30 minutes; the engine restarts several times
docker compose start edge0
```

The result is saved in the model folder as `strata-tuning.json`, and every start applies it (the
log says `edge0: tuned for this machine on <date>: ...`). Nothing outside the model folder is read
or written. A tune is saved per setup — GPU, VRAM, CPU, RAM, model, `MODEL_OPTIONS` and `CONTEXT` —
so after changing any of those the log says there is no tune for this setup, and you can run `tune`
again; the earlier entries stay in the file. A flag in `STRATA_ARGS` overrides the same tuned flag,
and `TUNING=off` ignores the file.

With Portainer, stop the stack and run the tune from a shell with the same image, model folder and
settings, for example:

```sh
docker run --rm --gpus all --ulimit memlock=-1:-1 --shm-size 2g \
  -v /path/to/models:/models -e EDGE0_MODEL=Edge0-145B-q3k edge0-infer:cuda13 tune
```

### Engine tuning

The words of `STRATA_ARGS` are appended to the engine's command line, after the model template's
own flags, so write flags and values separated by spaces. A flag the template already sets (such as
`--kv`) is refused; use `CONTEXT` instead of `--max-context`. Example from the Edge0 docs, for
tuning to one machine:

```sh
STRATA_ARGS=--pool-workers 24 --pcie-mode dma --pcie-frac 0.1 --vram-reserve-mib 700 --kv-resident 32768
```

| Flag | Description |
|---|---|
| `--vram-reserve-mib N` | GPU memory to leave free (default 700). Raise it if other programs use the GPU. |
| `--kv-resident N` | Tokens of KV cache kept on the GPU. |
| `--pool-workers N` | CPU worker threads. |
| `--pcie-mode`, `--pcie-frac` | How expert transfers over PCIe are scheduled. |

`--max-context` is rejected here (use `CONTEXT`), as are flags the model's template already sets.

## Running without Compose

```sh
docker run -d --name edge0 --gpus all \
  --ulimit memlock=-1:-1 --shm-size 2g \
  -p 8080:8080 \
  -v /path/to/models:/models \
  -e API_KEY=change-me -e EDGE0_MODEL=Edge0-145B-q3k \
  edge0-infer:cuda13
```

Any flags after the image name are added to Strata's server (`serve/server.py`). A command that doesn't start with
`-` runs instead of the server. For example, to check that a model folder is complete:

```sh
docker run --rm -v /path/to/models:/models edge0-infer:cuda13 \
  edge0-infer inspect --model /models/Edge0-145B-q3k --json
```

The entrypoint also accepts `MODEL_DIR` (a full path, instead of `EDGE0_MODEL`) and
`API_KEY_FILE` (a file holding the key, e.g. a Docker secret, instead of `API_KEY`).

## Building

```sh
docker compose build
# or
docker build -t edge0-infer:cuda13 .
```

The build follows the upstream instructions (`edge0-strata/EDGE0.md`, "Build (Linux, NVIDIA)"):
it fetches the pinned llama.cpp commit that provides ggml, builds the `strata` engine with
`-DSTRATA_ENABLE_CUDA=ON`, creates the engine's Python environment, and builds edge0-infer with its
`cpu` preset. edge0-infer needs no GPU backend for these models because edge0-strata does the GPU
work. The final image is based on `nvidia/cuda:13.0.0-runtime` and holds only what runs.

Build arguments:

| Argument | Default | Description |
|---|---|---|
| `CUDA_ARCHITECTURES` | `120` | Compute capabilities to compile for, separated by `;`. |
| `EDGE0_REF` | `main` | Upstream branch, tag or commit. Pin a commit for reproducible builds. |
| `EDGE0_REPO` | upstream GitHub URL | Use a fork. |
| `LLAMA_CPP_COMMIT` | `3cf0325…` | ggml source for the engine. If you change `EDGE0_REF`, use the commit that `edge0-strata/EDGE0.md` names at that revision. |
| `CUDA_VERSION` | `13.0.0` | CUDA base image version. CUDA 13 is required. |

### Building for other GPUs

Find your GPU's compute capability with
`nvidia-smi --query-gpu=compute_cap --format=csv` (12.0 → `120`) and build for it:

```sh
docker build -t edge0-infer:cuda13 --build-arg CUDA_ARCHITECTURES="89" .
```

Common values: `120` RTX 50-series, `89` RTX 40-series / L4 / L40, `86` RTX 30-series / A10,
`90` H100, `80` A100. Several, e.g. `"86;89;120"`, make one image for all of them, but each one
adds build time and image size. Only `120` has been tested with this image.

### Publishing

```sh
docker tag edge0-infer:cuda13 yourname/edge0-infer:cuda13
docker push yourname/edge0-infer:cuda13
```

Then set `EDGE0_IMAGE=yourname/edge0-infer:cuda13` so Compose pulls it instead of building.

## Using with Portainer

Create a stack from this repository (Repository build method, compose path `compose.yaml`), or
paste `portainer-stack.yaml` into the web editor and add the variables above under **Environment
variables**. `portainer-stack.yaml` is `compose.yaml` without the build section, because the web
editor has no build context: build or pull the image first and set `EDGE0_IMAGE` if it has
another name.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `edge0: set API_KEY or API_KEY_FILE` | No key configured. Set `API_KEY`. |
| `edge0: no model.json in /models/...` | `MODELS_DIR` or `EDGE0_MODEL` doesn't point at a model folder. The folder named by `EDGE0_MODEL` must directly contain `model.json`. |
| `401` / `Invalid API key` | Client sent no key or the wrong one. In the web page, enter the `API_KEY` value when it asks (it's kept in that browser). |
| Web page loads but chat gets `403` | The page was opened through an address the server doesn't know (reverse proxy or tunnel). Add it to `TRUSTED_ORIGINS`. |
| `edge0: ... holds model '...'; supported: ...` | The folder's `model.json` names a model this image has no template for. |
| Container restarts while loading, or out-of-memory errors | The model doesn't fit this GPU/RAM. Use the `-q3k` folder, lower `CONTEXT`, try `MODEL_OPTIONS=du=q4`, or close other programs using the GPU. |
| `no kernel image is available for execution on the device` | The image wasn't built for your GPU. Rebuild with your `CUDA_ARCHITECTURES`. |
| Container is `unhealthy` | The model took longer than 30 minutes to load, or the server stopped. Check `docker compose logs`. |
| Second request waits | Expected: requests are queued and answered one at a time. |

Known limitation from upstream: temperature 0 isn't bit-for-bit repeatable.

## Licenses

The Dockerfile, Compose file and scripts in this repository: Apache License 2.0, see [`LICENSE`](LICENSE). The image contains Edge0
(Apache-2.0, see `/opt/edge0/LICENSE` and `/opt/edge0/NOTICE` in the image) and the third-party
works listed in its NOTICE, plus NVIDIA CUDA runtime components under NVIDIA's license. Model
weights are licensed separately; see each model card.
