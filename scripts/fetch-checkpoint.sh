#!/usr/bin/env bash
# Usage: scripts/fetch-checkpoint.sh [--delete-weights] <dir> <bundle>
#
# Checks what `iris compile` and the reference read of one bundle in <dir> against the pinned
# sha256s, downloading what is missing through huggingface_hub with hf_xet (a single curl
# connection to huggingface.co runs at about 1.6 MB/s, hf_xet at about 45 MB/s): text 8.9 GB (the
# language model of Qwen/Qwen3-VL-4B-Instruct), base 12 GB, turbo 6 GB, depth 12 GB and upscaler
# 12 GB (speridlabs/iris-3b). --delete-weights deletes that bundle's weights (*.safetensors, the
# empty prompts kept), the small files staying.
set -euo pipefail
DELETE_WEIGHTS=0
if [ "${1:-}" = "--delete-weights" ]; then
  DELETE_WEIGHTS=1
  shift
fi
DIR="$1"
BUNDLE="$2"
TEXT_REPOSITORY="Qwen/Qwen3-VL-4B-Instruct"
TEXT_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
IRIS_REPOSITORY="speridlabs/iris-3b"
IRIS_REVISION="adc05b909fb05d4276d26aae5a52b36406b2ef26"
# <bundle> <sha256> <local path under the directory>, each local path text/<file> or iris/<file>.
FILES="
text 30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9 text/model-00001-of-00002.safetensors
text 046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6 text/model-00002-of-00002.safetensors
text 58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b text/model.safetensors.index.json
text edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d text/config.json
text 8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44 text/generation_config.json
text a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7 text/tokenizer.json
text c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399 text/tokenizer_config.json
text ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 text/vocab.json
text 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 text/merges.txt
text 6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4 text/chat_template.json
base a90e1f4c197bac8491f82078def68884b6a3527ea7fc97a693e3e367372aa80f iris/model.safetensors
base 077ce19156d9425a7e665a0059be02b3c5a1fcd7aea7607047b4f7486efa6cd5 iris/config.yaml
turbo 7148dbba3117567e34c86fadaae6e10325feb2c0ff3f43fe4938c15c75b356e1 iris/turbo/model.safetensors
turbo 077ce19156d9425a7e665a0059be02b3c5a1fcd7aea7607047b4f7486efa6cd5 iris/turbo/config.yaml
turbo 7c6f607c7b9eb3d42c0caac9c3e5acfcacf71f8c89d7174a32079eb4d3a6b74f iris/turbo/distill.yaml
depth e7aabbd5992637914407bfb5541f67c727ae30516d63eb32a85c541f8e5d8952 iris/depth/model.safetensors
depth ea77e942f2ac7698bf4e5a2657d1820839b7e47212762ae31fbe5234beef33e6 iris/depth/empty_prompt.safetensors
depth f2a9fbdccabc1a27ba5475ed0ad4bfc35dc854d1008be6a167f8b3de8bfd4dcf iris/depth/config.yaml
upscaler b4235e5f60bd1c0cb761f13b9549e31aefd25e3814aded4349f1f156ce0abeaa iris/upscaler/model.safetensors
upscaler 3cef1162f07bb81f7fb37fc7c997a9305a000f4de2563e61982d5d373a61d225 iris/upscaler/empty_prompt.safetensors
upscaler da09cc9a84c6ed6014f9c788640ee24bca0b79c1f9fa55b947a52357ca32a51d iris/upscaler/config.yaml
"
SELECTED="$(awk -v bundle="$BUNDLE" '$1 == bundle { print $2 "  " $3 }' <<< "$FILES")"
if [ -z "$SELECTED" ]; then
  echo "no bundle $BUNDLE (text, base, turbo, depth, upscaler)" >&2
  exit 1
fi
if [ "$DELETE_WEIGHTS" = 1 ]; then
  while read -r _ path; do
    case "$path" in
      *empty_prompt.safetensors) ;;
      *.safetensors) rm -f "$DIR/$path" ;;
    esac
  done <<< "$SELECTED"
  exit 0
fi
mkdir -p "$DIR"
listing="$(mktemp)"
trap 'rm -f "$listing"' EXIT
while read -r _ path; do
  if [ -s "$DIR/$path" ]; then
    continue
  fi
  case "$path" in
    text/*) echo "$TEXT_REPOSITORY $TEXT_REVISION text ${path#text/}" >> "$listing" ;;
    iris/*) echo "$IRIS_REPOSITORY $IRIS_REVISION iris ${path#iris/}" >> "$listing" ;;
  esac
done <<< "$SELECTED"
if [ -s "$listing" ]; then
  uv run --no-project --with huggingface_hub==0.36.2 --with hf_xet==1.8.0 python -I - \
    "$DIR" "$listing" <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download

root, listing = sys.argv[1:3]
with open(listing) as lines:
    for line in lines:
        repository, revision, local, name = line.split()
        print(f"downloading {repository}/{name}", flush=True)
        hf_hub_download(repository, name, revision=revision, local_dir=os.path.join(root, local))
PY
fi
rm -rf "$DIR/text/.cache" "$DIR/iris/.cache"
(cd "$DIR" && shasum -a 256 -c --quiet - <<< "$SELECTED")
