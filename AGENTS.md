# AGENTS.md

Rules for coding agents in this repository that are its own. fusor-ai's
`vendor/fusor-ai/AGENTS.md` is the standard and binds here too (`CLAUDE.md`
imports both); this file adds what is particular to this product.

iris is the Iris-3B app: speridlabs' pixel-space text-to-image model
(speridlabs/iris-3b) with its four-step Turbo, its depth estimator and its
four-times restorer, and Qwen3-VL-4B-Instruct reading the prompts, running in
the browser on the CPU, sealed from the network, the image forming step by
step. The README describes what it does; don't invent scope, features or
options beyond the task.

## Repository

- `vendor/fusor-ai/models/iris/runtime`: the wasm-bindgen surface of the model,
  two runtimes of one model (`TextRuntime`, which turns a prompt into
  Qwen3-VL's states, and `ImageRuntime`, which turns those states, or a
  photo, into an image step by step), each built as four browser runtimes.
- `vendor/fusor-ai/models/iris/cli`: the native CLI `iris` (`compile`,
  `generate`, `depth`, `upscale`, `bench`, `parity`).
- `app/`: the page, a [fusor](https://github.com/fusor-rs/fusor) app with its own
  Cargo workspace: the form, the loading and the live measurements as Rust
  signals in `src/app.rs`, the template in `web/index.html` on fusor-ai's
  `packages/ui/base.css`, the engine bridge, the canvases, the photo decoding
  and the PNG downloads in `web/app.js`, and in `engine/` the two workers
  (`text.js`, `image.js`) on fusor-ai's worker kit, beside the runtimes `just
  runtimes` builds.
- `scripts/`: the checkpoint download (pinned to revisions, sha256-checked),
  the Node smoke test, the site bundle and the README's images;
  `e2e/run.mjs`: the headless Chrome run on fusor-ai's e2e kit.
- [fusor-ai](https://github.com/fusor-rs/fusor-ai): the model (`iris`), its
  compiler (`iris-compile`), the kernels, the blocks (`transformer`, `dit`,
  `dpm`, `flow`) and the bundle format come from `vendor/fusor-ai`, a git
  submodule pinned to a commit pushed to fusor-ai, with the browser loader and
  compute pool, the runtime build, the import check and the dev server.

What belongs where: model code (dimensions, tensor names, wiring, the prompt
template, the samplers' settings) and anything another model could reuse go to
fusor-ai, following its `AGENTS.md`. This repository holds what turns the model
into an app: the page, the checks against the reference and the deployment. A
change to fusor-ai lands there first; this repository never patches it.

Constraints that hold everywhere:

- fusor-ai comes only from `vendor/fusor-ai`, never a crates.io version, a
  `[patch]` table or a sibling checkout. The submodule moves to another pushed
  fusor-ai commit in a change of its own, with the parity numbers and the e2e
  result it gave. No copies of fusor-ai code here.
- The page owns every capability: reading and decoding photos, drawing on the
  canvases, encoding the PNG downloads, caching the bundles (OPFS) and
  measuring itself. The runtimes receive a request (and a photo's pixels) and
  return text states, previews and RGBA images; they do no I/O.
- The model runs as two runtime instances, each in its own worker and memory,
  because Qwen3-VL and a transformer together do not fit in one wasm32
  memory; the text states cross from one to the other as bytes through the
  page. Only one transformer is in memory at a time: choosing a mode that
  runs another one ends the image worker and starts a new one. Each instance's
  shared memory may grow to 4 GiB (`MAX_MEMORY` in the justfile): the text
  bundle (3.1 GB) with its activations, or a transformer bundle (2.6 GB) with
  its activations at 896 × 1344 (the largest size it generates) and a restoration's planes (a work image of
  up to 2048 × 2048 pixels). A change that grows the activations says by how
  much.
- Generation matches Iris-3B's reference (its own code, pinned in the
  justfile) stage by stage up to the int8 attention and the quantized
  weights; images of the same seed differ from the reference's, whose noise
  comes from torch's generator. Say so where a comparison could suggest
  otherwise.
- `just bundles` compiles the bundles that `dist/model` lacks; delete a
  bundle's directory to compile it again after a change to the compiler.
- The e2e compares the page's images byte for byte with the same build's
  single-threaded runtime in Node, and `MIN_SPREAD` (every image a picture) is
  never loosened.
- The models are Apache-2.0. The page, the README and the Space credit
  speridlabs and Qwen, with links.

## Checks

A change is done only when this passes:

```sh
just check
```

It runs formatting and Clippy on the page and syntax checks of the scripts;
fusor-ai's `just check` covers the runtimes and the CLI. A change to the
runtimes, the page, the CLI or fusor-ai also needs `just runtimes`, `just
smoke`, `just parity` (the same numbers as before unless the change is meant to
move them) and `just e2e`; a change meant to move speed also needs `just
bench`. The model is checked stage by stage against the reference goldens by
`iris parity`; extend it rather than adding a parallel harness.

## Commits

- Don't commit, push, tag, release or deploy unless asked. Deploying is
  described in the justfile (`deploy-model`, then `deploy-space`) and is done by
  the maintainer.
