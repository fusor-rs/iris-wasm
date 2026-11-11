<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/banner-dark.png">
    <img src="assets/brand/banner.png" alt="fusor-ai × Iris-3B: iris" width="640">
  </picture>
</p>

# Images, depth and restoration in the browser, on the CPU

Describe an image and watch it form, estimate a photo's depth, or enlarge a photo four times, all with [Iris-3B](https://huggingface.co/speridlabs/iris-3b), speridlabs' open model that draws in pixels rather than latents, computed on your own device. [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) reads the prompts. No server, no API key, no GPU required, and once the model has loaded, nothing you type or choose leaves the page.

iris is the app around the model: a [fusor](https://github.com/fusor-rs/fusor) page, the browser runtimes and a native CLI. The models, compiled to WebAssembly, are built with [fusor-ai](https://github.com/fusor-rs/fusor-ai), the toolkit for AI runtimes in Rust and WebAssembly, from the published weights.

**In development · v0.1**: this is experimental software. See [status and known limitations](#status-and-known-limitations).

[Run it](#run-it-locally) · [How it works](#how-it-works) · [How fast is it](#how-fast-is-it) · [Status](#status-and-known-limitations)

## Run it locally

You need Rust 1.99 or newer, installed with [rustup](https://rustup.rs), Node.js 22 or newer and [`just`](https://github.com/casey/just). Then:

```sh
git clone --recurse-submodules https://github.com/fusor-rs/iris-wasm && cd iris-wasm
just ui
```

and open the address it prints. `just ui` installs the pinned tools, downloads the checkpoints at pinned revisions (51 GB, each file sha256-checked), compiles them into five bundles (3.1 GB for the text encoder, 2.6 GB for each of the four transformers), builds the browser runtimes and the page, and serves it on `http://localhost:8787` with the bundles beside it. `just bundles` does the same compile one checkpoint at a time, deleting each checkpoint's weights once its bundle is written, for a small disk.

The page has four modes:

- **Turbo** (the default) draws the prompt in four steps. Choose a size and a seed.
- **Generate** runs the base model with classifier-free guidance, at the model card's 100 steps and guidance 3 unless you change them; shift, a negative prompt and the seed are yours too. The page shows the time a run should take.
- **Depth** reads a photo you choose or drop on the page and paints its depth map.
- **Upscale** enlarges a photo four times, one 1024 × 1024 tile at a time.

Turbo and Generate draw at the seven sizes Iris-3B was trained at, about one megapixel each: 1024 × 1024 (the default), 896 × 1200, 896 × 1344, 752 × 1344 and their landscape twins. The engine refuses any other size.

Write a prompt and press **Generate**, even while the model is still loading: the request waits and starts the moment the model is ready. After every step the page draws the clean image that step predicts; the finished image is yours to download as a PNG. Every job is measured live: the time, the seconds per step or tile, the steps done, and the network requests since the model loaded, which stay at zero. Choosing another mode loads that mode's transformer (from this device when it has been loaded before) and lets go of the previous one.

## More ways to run it

### Build everything from source

```sh
just build    # also downloads the checkpoints and compiles the bundles
just serve
```

`CHECKPOINT=<dir>` reuses checkpoints you already have, and `PROFILE=f32` compiles the full-precision bundles that `just parity-f32 <bundle>` checks against the reference.

### The CLI

The CLI works without a browser:

```sh
just generate --prompt "a red fox in the snow" --variant base --steps 30 --seed 7
just generate --prompt "a lighthouse at dusk" --width 752 --height 1344
just depth photo.png
just upscale photo.png --short-side 256 --long-side 256
```

`iris bench` times each stage and `iris parity` compares every stage with the reference's outputs.

## How it works

- **Two runtimes, two memories.** A WebAssembly memory holds at most 4 GiB, and Qwen3-VL-4B (the text encoder) and a 3-billion-parameter transformer together do not fit in one. The page runs them as two instances in two workers; the encoder's output, the 12 layers' states of the prompt's (and, for guidance, the negative prompt's) rows, crosses to the image instance as bytes through the page. Only one transformer is in memory at a time.
- **4 and 7 bits.** The bundles store the feed-forward matrices of Qwen3-VL and of the transformers' blocks in 4 bits and every other large matrix in 7, with a scale per 32 values (fusor-ai's `Q4b32` and `Q7b32`); the matrix multiply quantizes activations to int8 on the fly and adds up each group's products as integers.
- **Pixels, not latents.** Iris-3B has no autoencoder: its transformer reads 16 × 16 patches of the image itself, with a small per-pixel head, so a preview after every step is the model's own prediction of the clean image.
- **The samplers.** The base model integrates with DPM-Solver++(2M) at shifted flow-matching noise levels, steering the prompt's prediction by the negative prompt's (classifier-free guidance). Turbo takes four Euler steps over its distilled levels. Depth is one pass. The restorer enlarges the photo with a bicubic filter, then repairs it tile by tile with one pass over each 1024 × 1024 tile, fusing the overlaps.
- **Checked against the reference.** Every stage agrees with Iris-3B's own code in full precision; the quantized bundles are checked by `iris parity` against the same goldens, and the differences that remain are the 4- and 7-bit weights and the int8 attention.
- **Threads and SIMD where the browser has them.** The page picks one of four builds: worker threads when it is cross-origin isolated, relaxed SIMD when the engine supports it, plain SIMD128 everywhere else. The page's images equal, byte for byte, those of the same build run single-threaded in Node.
- **Downloaded once, verified always.** The bundles stream in parts, each checked against its BLAKE3 hash, and stay in the browser's origin-private storage, so the next visit loads from this device.
- **Sealed.** The runtimes import no network, storage, clock or entropy capability; fusor-ai's import check proves it on every build. The page counts its network requests while it works, and the count is zero.
- **A fusor page.** The form, the loading and the measurements are Rust signals with HTML templates; a small JavaScript module runs the engine workers, reads the photos and draws on the canvases.

## How fast is it

Measured on an Apple M4 Pro, eight threads, with other work loading the machine (the figures are from `just integrate`):

- **In Chrome**, one pass of the transformer over 1024 × 1024 pixels takes 18.6 s with relaxed SIMD and 66 s with plain SIMD128 (that run shared the machine with its single-threaded references). A Turbo image at 1024 × 1024 takes 75 s; a guided step of the base model takes 37 s, so its default 100 steps take about an hour. A restored 1024 × 1024 tile takes 18 s and a depth map of a small photo 2 s. The first visit loaded Turbo's 5.7 GB of bundles from the local server in 14 s; the next visit loads them from the browser's storage in 6 s.
- **Natively**, the CLI's pass at 1024 × 1024 takes 24 s, a Turbo image 97 s, reading a prompt 0.33 s.

## Status and known limitations

iris is v0.1 and experimental.

### Platform status

| Platform or capability | Status |
| --- | --- |
| Chrome and Edge, threads and relaxed SIMD | Headless Chrome end-to-end check (`just e2e`): Turbo, Generate, Depth and Upscale, each previewed where it has steps, the image equal to the same build in Node, the PNG download, revisit |
| Browsers without relaxed SIMD (Safari, older Firefox) | SIMD128 builds, checked in Chrome with threads and in Node single-threaded (`just smoke`); not tested in those browsers |
| Native CLI (macOS on Apple silicon) | Built and measured; no release archives yet |
| Phones and small machines | Not supported: the two instances need about 7 GB of memory in one tab (an estimate from the memory arithmetic, not a measurement) |

**Not supported yet:**

- Sizes other than the seven Iris-3B was trained at: the engine refuses them rather than draw an image the model was not made for.
- The base model's default of 100 guided steps is about an hour at 1024 × 1024 on an 8-thread laptop; fewer steps are faster but lose detail.
- The first download is 5.7 GB for Turbo, and 2.6 GB more for each other mode.
- Restorations up to 2048 × 2048 pixels: the photo is shrunk to the chosen budget (at most 512 px a side) before it is enlarged, where the reference allows 512 × 1024.
- Prompts are cut at 295 tokens, as the reference cuts them.
- A GPU backend.

The models are [Iris-3B](https://huggingface.co/speridlabs/iris-3b) by speridlabs and [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) by Qwen, both under the Apache-2.0 license; their license and notice ship with the bundles as `model/LICENSE` and `model/NOTICE`.

## How is this different from speridlabs' own code?

Iris-3B's [model card](https://huggingface.co/speridlabs/iris-3b) runs it with Python and PyTorch on an NVIDIA GPU. iris is a separate implementation on fusor-ai: the same weights and the same samplers' maths, its own kernels, checked stage by stage against the reference's modules, running in the browser on the CPU. The noise comes from fusor-ai's generator, not torch's, so a seed gives a different image than the reference's. It is far slower than a GPU; it needs no GPU and no server.

## Contributing

iris is an open source project in early development. The page lives in `app/`; the runtimes and the CLI live in fusor-ai's `models/iris`; the model and its toolkit come from fusor-ai, its crates and browser scripts from the `vendor/fusor-ai` submodule. [AGENTS.md](AGENTS.md) is the code standard every change follows, and `just check` its check.

iris is built with the help of AI tools, mainly Claude Code, under the same rules the project asks of contributors.

## License

Licensed under the [MIT License](LICENSE).
