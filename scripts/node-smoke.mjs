// Runs a single-threaded wasm runtime build in Node against the bundles, each runtime in its own
// worker thread (its own module instance and memory), one after another: the text runtime
// encodes the prompt and the negative prompt; Turbo makes an image, the base model one with
// guidance, the depth model reads a photo and the restorer restores one, each step by step with
// every preview checked; the time of each stage is printed. Fails if a preview is missing, an
// image has the wrong size or is blank. With `--out <dir>`, writes `<mode>.rgba` per mode. The
// requests' fields are options (the e2e run takes its references from here). `--stop-after <n>`
// ends a generation after its first n steps, its last preview taken as its image.
//
//   node scripts/node-smoke.mjs dist/model --build web-single [--modes turbo,base,depth,upscaler]
//     [--prompt text] [--negative text] [--width 1024] [--height 1024] [--steps 2] [--guidance 3]
//     [--shift 4] [--seed 42] [--depth-photo fixtures/media/depth.png] [--max-side 256]
//     [--upscale-photo fixtures/media/upscale.png] [--budget 256] [--stop-after n] [--out dir]

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { inflateSync } from 'node:zlib';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const PROMPT = 'A lighthouse on a rocky coast at sunset, waves crashing';
const CHUNK = 8 << 20;
const RGBA = 4;
const MIN_COLOURS = 2;
// Turbo reads only the size and the seed; the page sends these for the rest.
const TURBO = { steps: 4, guidance: 1, shift: 4 };
// The modes, by the bundle that runs them, and the runtime call that starts each.
const STARTS = { turbo: 'start_generation', base: 'start_generation', depth: 'start_depth', upscaler: 'start_upscale' };

const seconds = (since) => ((performance.now() - since) / 1000).toFixed(2);

/** The class `className` of the runtime build, with the bundle at `bundle` pushed into it. */
async function load(className, bundle, build) {
  const dist = new URL(`../app/engine/dist/${build}/`, import.meta.url);
  const glue = await import(new URL('iris_runtime.js', dist));
  glue.initSync({ module: readFileSync(new URL('iris_runtime_bg.wasm', dist)) });
  const t0 = performance.now();
  const manifest = readFileSync(join(bundle, 'manifest.json'), 'utf8');
  const runtime = new glue[className](manifest);
  let ready = false;
  JSON.parse(manifest).parts.forEach((part, index) => {
    const bytes = readFileSync(join(bundle, part.file));
    for (let offset = 0; offset < bytes.length; offset += CHUNK) runtime.write_part(index, bytes.subarray(offset, offset + CHUNK));
    ready = runtime.end_part(index);
  });
  if (!ready) throw new Error(`${className} not ready after every part`);
  console.log(`${build} ${className} ${bundle}: load ${seconds(t0)}s`);
  return runtime;
}

/** In a worker thread: the states of the prompt, and of the negative prompt when `negative` is a string. */
async function encode({ bundle, build, prompt: promptText, negative: negativeText }) {
  const runtime = await load('TextRuntime', join(bundle, 'text'), build);
  const t0 = performance.now();
  const prompt = runtime.encode(promptText);
  const negative = negativeText === null ? new Float32Array(0) : runtime.encode(negativeText);
  parentPort.postMessage({ prompt, negative, seconds: seconds(t0) }, [prompt.buffer, negative.buffer]);
}

/** The colours an image has, up to `MIN_COLOURS`. */
function colours(image) {
  const seen = new Set();
  for (let at = 0; at < image.length && seen.size < MIN_COLOURS; at += RGBA) seen.add((image[at] << 16) | (image[at + 1] << 8) | image[at + 2]);
  return seen.size;
}

/** The request of `mode` as the page sends its JSON; the seed is a JSON number, however long. */
function requestFor(mode, values) {
  if (mode === 'depth') return `{"max_side":${values.maxSide}}`;
  if (mode === 'upscaler') return `{"short_side":${values.budget},"long_side":${values.budget}}`;
  const { steps, guidance, shift } = mode === 'turbo' ? TURBO : values;
  return `{"width":${values.width},"height":${values.height},"steps":${steps},"guidance":${guidance},"shift":${shift},"seed":${values.seed}}`;
}

/** In a worker thread: one mode's job, step by step; resolves false when a check failed. */
async function image({ bundle, build, mode, values, states }) {
  const runtime = await load('ImageRuntime', join(bundle, mode), build);
  const photo = { depth: values.depthPhoto, upscaler: values.upscalePhoto }[mode];
  const { rgba, width, height } = photo ? readPng(readFileSync(photo)) : {};
  const inputs = photo ? [rgba, width, height] : [states.prompt, states.negative];
  let t0 = performance.now();
  const plan = JSON.parse(runtime[STARTS[mode]](requestFor(mode, values), ...inputs));
  console.log(`  ${mode} start: ${seconds(t0)}s`);
  const generates = !photo;
  const stopAfter = generates && values['stop-after'] ? Number(values['stop-after']) : Infinity;
  let failed = false;
  let preview = null;
  for (let done = false, step = 0; !done && step < stopAfter; ) {
    t0 = performance.now();
    const progress = JSON.parse(runtime.step());
    ({ done, step } = progress);
    preview = generates && !done ? runtime.preview() : null;
    failed ||= preview !== null && preview.length !== plan.width * plan.height * RGBA;
    console.log(`  ${mode} step ${progress.step}/${progress.steps}: ${seconds(t0)}s`);
  }
  const pixels = preview ?? runtime.take_image();
  if (values.out) writeFileSync(join(values.out, `${mode}.rgba`), pixels);
  const wrongSize = pixels.length !== plan.width * plan.height * RGBA;
  const blank = colours(pixels) < MIN_COLOURS;
  console.log(`  ${mode}: ${plan.width}×${plan.height}, ${blank ? 'blank' : 'varied'}`);
  parentPort.postMessage({ failed: failed || wrongSize || blank });
}

/** Runs `data` in a worker thread of this script, which ends with its result; its output is printed first. */
async function inWorker(data) {
  const worker = new Worker(fileURLToPath(import.meta.url), { workerData: data });
  const exited = new Promise((resolve) => worker.once('exit', resolve));
  const result = await new Promise((resolve, reject) => {
    worker.once('message', resolve);
    worker.once('error', reject);
  });
  await exited;
  return result;
}

async function main() {
  const { values: options, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      build: { type: 'string', default: 'web-single' },
      modes: { type: 'string', default: Object.keys(STARTS).join(',') },
      prompt: { type: 'string', default: PROMPT },
      negative: { type: 'string', default: '' },
      width: { type: 'string', default: '1024' },
      height: { type: 'string', default: '1024' },
      steps: { type: 'string', default: '2' },
      guidance: { type: 'string', default: '3' },
      shift: { type: 'string', default: '4' },
      seed: { type: 'string', default: '42' },
      'depth-photo': { type: 'string', default: 'fixtures/media/depth.png' },
      'max-side': { type: 'string', default: '256' },
      'upscale-photo': { type: 'string', default: 'fixtures/media/upscale.png' },
      budget: { type: 'string', default: '256' },
      'stop-after': { type: 'string' },
      out: { type: 'string' },
    },
  });
  const bundle = positionals[0] ?? 'dist/model';
  const modes = options.modes.split(',');
  const unknown = modes.filter((mode) => !(mode in STARTS));
  if (unknown.length > 0) throw new Error(`no mode ${unknown.join(', ')}: ${Object.keys(STARTS).join(', ')}`);
  if (!/^\d+$/.test(options.seed)) throw new Error('--seed is a whole number');
  if (options.out) mkdirSync(options.out, { recursive: true });
  const values = { ...options, maxSide: options['max-side'], depthPhoto: options['depth-photo'], upscalePhoto: options['upscale-photo'], guidance: Number(options.guidance) };

  let states = null;
  if (modes.some((mode) => mode === 'turbo' || mode === 'base')) {
    const guided = modes.includes('base') && values.guidance > 1;
    states = await inWorker({ stage: 'text', bundle, build: options.build, prompt: options.prompt, negative: guided ? options.negative : null });
    console.log(`  prompts: ${states.seconds}s`);
  }
  let failed = false;
  for (const mode of modes) {
    const result = await inWorker({ stage: 'image', bundle, build: options.build, mode, values, states });
    failed ||= result.failed;
  }
  if (failed) process.exit(1);
}

if (isMainThread) await main();
else if (workerData.stage === 'text') await encode(workerData);
else await image(workerData);

/** The pixels of an 8-bit, non-interlaced PNG (grey, grey and alpha, RGB or RGBA), as RGBA. */
function readPng(bytes) {
  const CHANNELS = { 0: 1, 4: 2, 2: 3, 6: 4 };
  let width = 0;
  let height = 0;
  let channels = 0;
  const data = [];
  for (let at = 8; at < bytes.length; ) {
    const size = bytes.readUInt32BE(at);
    const type = bytes.toString('ascii', at + 4, at + 8);
    const body = bytes.subarray(at + 8, at + 8 + size);
    if (type === 'IHDR') {
      [width, height] = [body.readUInt32BE(0), body.readUInt32BE(4)];
      if (body[8] !== 8 || body[12] !== 0 || !(body[9] in CHANNELS)) throw new Error('an 8-bit, non-interlaced PNG expected');
      channels = CHANNELS[body[9]];
    }
    if (type === 'IDAT') data.push(body);
    at += 12 + size;
  }
  const raw = inflateSync(Buffer.concat(data));
  const stride = width * channels;
  const pixels = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const left = x >= channels ? pixels[y * stride + x - channels] : 0;
      const up = y > 0 ? pixels[(y - 1) * stride + x] : 0;
      const corner = x >= channels && y > 0 ? pixels[(y - 1) * stride + x - channels] : 0;
      pixels[y * stride + x] = (line[x] + predict(filter, left, up, corner)) & 0xff;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const pixel = pixels.subarray(i * channels, (i + 1) * channels);
    const [r, g, b] = channels >= 3 ? pixel : [pixel[0], pixel[0], pixel[0]];
    rgba.set([r, g, b, channels % 2 === 0 ? pixel[channels - 1] : 255], i * 4);
  }
  return { width, height, rgba };
}

/** PNG's filter predictor of a byte from its left, upper and upper-left neighbours. */
function predict(filter, left, up, corner) {
  if (filter === 0) return 0;
  if (filter === 1) return left;
  if (filter === 2) return up;
  if (filter === 3) return (left + up) >> 1;
  const estimate = left + up - corner;
  const [toLeft, toUp, toCorner] = [Math.abs(estimate - left), Math.abs(estimate - up), Math.abs(estimate - corner)];
  return toLeft <= toUp && toLeft <= toCorner ? left : toUp <= toCorner ? up : corner;
}
