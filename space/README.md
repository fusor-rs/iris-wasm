---
title: iris-wasm
short_description: Images, depth and 4× restoration on your own CPU
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
  - fusorlabs/iris-3b-web
  - speridlabs/iris-3b
  - Qwen/Qwen3-VL-4B-Instruct
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Text to image, depth estimation and four-times restoration with speridlabs'
[Iris-3B](https://huggingface.co/speridlabs/iris-3b), computed on your own CPU in the browser: the
models, compiled to WebAssembly with fusor-ai, download once and then run sealed from the network.
Describe an image and watch it form in four steps with Turbo or up to a hundred with the base
model, at one of the seven sizes of about a megapixel Iris-3B was trained at, estimate a photo's
depth, or enlarge a photo four times. Qwen3-VL-4B-Instruct reads the prompt. It runs as
two instances, one for the text encoder and one for the transformer of the chosen mode, so each
stays under the 4 GiB a WebAssembly memory can hold; your prompts and photos never leave the page.
The models are Apache-2.0; their license and notice ship beside the bundles as `model/LICENSE` and
`model/NOTICE`. The page's code is MIT; its source is
[github.com/fusor-rs/iris-wasm](https://github.com/fusor-rs/iris-wasm).
