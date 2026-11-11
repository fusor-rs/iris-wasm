---
license: apache-2.0
base_model:
  - speridlabs/iris-3b
  - Qwen/Qwen3-VL-4B-Instruct
pipeline_tag: text-to-image
library_name: fusor-ai
tags:
  - webassembly
  - wasm
  - on-device
  - text-to-image
  - depth-estimation
  - image-to-image
  - super-resolution
---

# Iris-3B, compiled for the browser

speridlabs' [Iris-3B](https://huggingface.co/speridlabs/iris-3b), a pixel-space text-to-image model
with a four-step Turbo, a depth estimator and a four-times restorer, compiled with the Qwen3-VL-4B
text encoder it reads its prompts through into five [fusor-ai](https://github.com/fusor-rs/fusor-ai)
bundles: the weights the browser runtime of
[iris-wasm](https://github.com/fusor-rs/iris-wasm) loads to turn a prompt or a photo into an image on
the device's CPU. Try it in the Space [fusorlabs/iris-wasm](https://huggingface.co/spaces/fusorlabs/iris-wasm).

- **Sources:** `speridlabs/iris-3b` at revision `adc05b909fb05d4276d26aae5a52b36406b2ef26` and
  `Qwen/Qwen3-VL-4B-Instruct` at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, compiled with
  the `mix` profile into five bundles, because one WebAssembly memory holds at most 4 GiB and the
  page runs the text encoder and one transformer at a time:
  - `text/`: Qwen3-VL's language model up to the last layer Iris reads, and its tokenizer; the
    feed-forward matrices 4-bit, every other large matrix 7-bit (3105 MB).
  - `base/`, `turbo/`, `depth/`, `upscaler/`: one transformer each (the base model, its four-step
    distillation, the depth estimator and the restorer); the feed-forward matrices of the blocks
    4-bit, the other large matrices 7-bit, the layers that read straight into or out of pixels in
    f32 (2558 MB each). The depth and the restorer bundles hold the empty prompt's states.
- **Format:** each `manifest.json` lists every tensor and part; each part is named by its BLAKE3
  hash, which the runtime checks as it loads.
- **Reproduce it:** `just bundles` in iris-wasm downloads the checkpoints and compiles the same
  bundles.

This is not a PyTorch or safetensors checkpoint: it only loads in the iris-wasm runtime.

The models are Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
