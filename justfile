# Iris-3B in the browser: text to image (Turbo and the base model), depth and 4× restoration.
# `just ui` builds everything, the models included, and serves the page; `just build` then
# `just serve` do the same in two steps.

import 'vendor/fusor-ai/just/product.just'

name := "iris"
# Each runtime instance's shared memory may grow to all of wasm32's 4 GiB: Qwen3-VL's 3.1 GB
# bundle, or a 2.6 GB transformer with its activations at 896 × 1344 and a restoration's planes.
export MAX_MEMORY := "4294967296"
checkpoint := env_var_or_default("CHECKPOINT", ".cache/checkpoint")
profile := env_var_or_default("PROFILE", "mix")
bundle_names := "text base turbo depth upscaler"
model_f32 := "dist/model-f32"
goldens := "fixtures/goldens"
# The reference the goldens come from (vendor/fusor-ai/models/iris/reference), the fixtures and the
# license and notice the bundles ship with: Iris-3B's code at a pinned commit, under pinned packages.
iris_repository := ".cache/iris-3b"
iris_commit := "13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e"
reference := "uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.19.0 --with accelerate==1.15.0 --with safetensors==0.8.0 --with numpy==2.5.4 --with omegaconf==2.3.0 --with pillow==12.3.0 --with matplotlib==3.11.2 python -I"

host:
    bash -n scripts/fetch-checkpoint.sh
    for file in app/web/app.js app/engine/text.js app/engine/image.js scripts/node-smoke.mjs scripts/site-bundle.mjs scripts/render-images.mjs e2e/run.mjs; do node --check "$file" || exit 1; done

repository:
    test -d {{iris_repository}} || git clone --quiet https://github.com/speridlabs/iris-3b.git {{iris_repository}}
    git -C {{iris_repository}} checkout --quiet {{iris_commit}}

# Every checkpoint (51 GB, sha256-checked, downloading what is missing) compiled into dist/model,
# the license and notice beside the bundles. The checkpoints stay: `just bundles` is the one that
# fits a small disk.
model: native repository
    for name in {{bundle_names}}; do scripts/fetch-checkpoint.sh {{checkpoint}} $name && just fusor_ai={{fusor_ai}} compile $name {{model}} {{profile}} || exit 1; done
    cp {{iris_repository}}/LICENSE {{iris_repository}}/NOTICE {{model}}/

# The shipped bundles that dist/model lacks, one checkpoint at a time, each checkpoint's weights
# deleted once its bundle is written: the 51 GB of weights beside the 13.4 GB of bundles do not fit
# a small disk (fetch-checkpoint.sh downloads them again when a recipe needs them). Delete a
# bundle's directory to compile it again.
bundles: native repository
    for name in {{bundle_names}}; do test -f {{model}}/$name/manifest.json || just fusor_ai={{fusor_ai}} bundle $name || exit 1; done
    cp {{iris_repository}}/LICENSE {{iris_repository}}/NOTICE {{model}}/

# One shipped bundle, its checkpoint's weights deleted after.
bundle name: native
    scripts/fetch-checkpoint.sh {{checkpoint}} {{name}}
    just fusor_ai={{fusor_ai}} compile {{name}} {{model}} {{profile}}
    scripts/fetch-checkpoint.sh --delete-weights {{checkpoint}} {{name}}

# One bundle compiled into `out` with `profile`.
compile name out profile: native
    rm -rf {{out}}/{{name}}
    target/release/iris compile {{checkpoint}} {{out}} --bundle {{name}} --profile {{profile}}

ui: model runtimes page serve

# Every bundle of dist/model against the reference's goldens (`--only <bundle>` narrows it).
parity *arguments: native
    target/release/iris parity {{model}} {{goldens}} {{arguments}}

# One bundle compiled f32 (a transformer's is 12 GB) against the goldens, then deleted; its
# checkpoint stays for `just bundle`.
parity-f32 name: native
    scripts/fetch-checkpoint.sh {{checkpoint}} {{name}}
    just fusor_ai={{fusor_ai}} compile {{name}} {{model_f32}} f32
    target/release/iris parity {{model_f32}} {{goldens}} --only {{name}}
    rm -rf {{model_f32}}/{{name}}

# One case's goldens from Iris-3B's reference in f32 on the CPU: `text` first, which `base` and
# `turbo` read.
goldens case: repository
    scripts/fetch-checkpoint.sh {{checkpoint}} {{case}}
    mkdir -p {{goldens}}
    {{nice}} {{reference}} {{fusor_ai}}/models/iris/reference/dump.py {{iris_repository}} {{checkpoint}} {{goldens}} {{case}}

# The native CLI on dist/model as it is: `just generate --prompt "a red fox in the snow" --variant base`.
generate *arguments: native
    target/release/iris generate {{model}} {{arguments}}

depth *arguments: native
    target/release/iris depth {{model}} {{arguments}}

upscale *arguments: native
    target/release/iris upscale {{model}} {{arguments}}

bench: native
    target/release/iris bench {{model}}

# The photos the smoke and the e2e read: two of Iris-3B's demo images.
fixtures: repository
    mkdir -p fixtures/media
    cp {{iris_repository}}/demo/examples/lowres/57.png fixtures/media/depth.png
    cp {{iris_repository}}/demo/examples/lowres/71.png fixtures/media/upscale.png
    printf '%s  %s\n' bcf61eabaa666cf58d610a302011a479ae1f91c376adef363c25ab5b42652ce8 fixtures/media/depth.png 2b2c5930051eb9b1c3293e41ebf13914aae81a20263e303ff97083e49cbdd103 fixtures/media/upscale.png | shasum -a 256 -c --quiet

images:
    cd {{fusor_ai}}/packages/e2e && npm ci --no-audit --no-fund --silent && npx playwright install chromium
    node scripts/render-images.mjs

site: runtimes page
    rm -rf target/site
    mkdir -p target/site
    cp -R {{app}}/dist/. target/site/
    find target/site -name '.fusor-*.json' -delete
    for name in {{bundle_names}}; do node scripts/site-bundle.mjs {{model}}/$name target/site/model/$name || exit 1; done
    cp {{model}}/LICENSE {{model}}/NOTICE target/site/model/

# The Space holds at most 1 GB, so it gets the page, the runtimes and the license and notice, and
# the page loads the bundles from the model repo fusorlabs/iris-3b-web (`just deploy-model`). The
# Space is created as a static Space first: creating it with the upload makes a Gradio Space,
# which Hugging Face refuses for a free organization.
# The site to the Hugging Face Space fusorlabs/iris-wasm, configured by space/README.md.
deploy-space: site
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf target/space
    rsync -a --exclude /model target/site/ target/space/
    rsync -a --include '*/' --include 'LICENSE*' --include 'NOTICE' --exclude '*' target/site/model/ target/space/model/
    cp space/README.md target/space/
    uv run --no-project --with huggingface_hub==0.36.2 python - target/space <<'PY'
    import sys
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo("fusorlabs/iris-wasm", repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        repo_id="fusorlabs/iris-wasm",
        repo_type="space",
        folder_path=sys.argv[1],
        delete_patterns="*",
        commit_message="Deploy iris-wasm {{version}}",
    )
    PY

# Deploy them before the Space whenever a bundle changes: a runtime only loads a bundle compiled
# with the same fusor-ai.
# The site's five bundles, with space/MODEL.md as their card, to the model repo
# fusorlabs/iris-3b-web.
deploy-model: site
    #!/usr/bin/env bash
    set -euo pipefail
    cp space/MODEL.md target/site/model/README.md
    uv run --no-project --with huggingface_hub==0.36.2 python - target/site/model <<'PY'
    import sys
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo("fusorlabs/iris-3b-web", exist_ok=True)
    api.upload_folder(
        repo_id="fusorlabs/iris-3b-web",
        folder_path=sys.argv[1],
        delete_patterns="*",
        commit_message="Deploy the iris-wasm {{version}} bundles",
    )
    PY
