// End-to-end run in headless Chrome, in a fresh profile.
//
// 1. Open the page. For each mode (Turbo, Generate, Depth, Upscale) make the reference: the same
//    build's single-threaded runtime in Node (`scripts/node-smoke.mjs`) runs the page's own
//    request: its prompt, negative prompt, guidance and shift, at the e2e's size, steps and seed,
//    or its photo. The references run in the background while the page works.
// 2. Turbo: press Generate while the model is still loading: the request is queued and runs the
//    moment both bundles are ready. The other modes load their transformer when chosen. Check
//    that a preview was drawn before the image (Turbo, Generate), that the image is a picture
//    (varied across the canvas), that it equals the reference's byte for byte, that the PNG
//    download exists, and that nothing was requested after the model was ready.
// 3. Reload: the revisit must load Turbo's bundles from OPFS.
//
//   PORT=8787 node e2e/run.mjs --build web-threaded [--threads 8]
//   node e2e/run.mjs --build web-threaded --url https://example.org   # a deployed page
//
// `--build` is the page's runtime, `web-threaded` or `web-threaded-relaxed`; the reference runs
// its single-threaded twin, `web-single` or `web-single-relaxed` (`just runtimes`), on dist/model.
// Results (JSON, the references' and the page's RGBA images and a screenshot) go to
// e2e/results/<timestamp>-*, or to `--out <dir>`.

import { spawn } from 'node:child_process';
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fetched, flags, run, STAMP } from '../vendor/fusor-ai/packages/e2e/kit.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const values = flags({});
const DEADLINE_MS = 30 * 60 * 1000;
const SIZE = { id: '1024x1024', width: 1024, height: 1024 };
const STEPS = 1;
const SEED = 42;
// The depth model sees the photo's own size (0); the restorer a photo shrunk to this many pixels a side.
const MAX_SIDE = 0;
const BUDGET = 256;
// An image's pixels spread over more than this standard deviation (0 to 255).
const MIN_SPREAD = 20;

// Each mode: the bundle that runs it (also the reference's name for it), the photo it reads, and
// the tiles of the page that the results keep as figures.
const CASES = [
  { mode: 'turbo', bundle: 'turbo', figures: { 'turbo-time': 'time', 'turbo-per-step': 'per-step' } },
  { mode: 'generate', bundle: 'base', figures: { 'base-per-step': 'per-step' } },
  { mode: 'depth', bundle: 'depth', photo: 'fixtures/media/depth.png', figures: { 'depth-time': 'time' } },
  { mode: 'upscale', bundle: 'upscaler', photo: 'fixtures/media/upscale.png', figures: { 'upscale-per-tile': 'per-step' } },
];

// The page's state machine is `main[data-phase]`: loading, ready, generating, broken.
const settled = () => /^(ready|broken)$/.test(document.querySelector('main')?.dataset.phase ?? '');
const loadedOrBroken = () => document.querySelector('main')?.dataset.loaded === 'true' || document.querySelector('main')?.dataset.phase === 'broken';
const loading = () => document.querySelector('main')?.dataset.loaded === 'false';

/** In the page: the canvas's size, its distinct colours and the spread of its pixels. */
function inspectCanvas() {
  const canvas = document.querySelector('#picture');
  const { data } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  const colours = new Set();
  let sum = 0;
  let squares = 0;
  for (let at = 0; at < data.length; at += 4) {
    colours.add((data[at] << 16) | (data[at + 1] << 8) | data[at + 2]);
    const grey = (data[at] + data[at + 1] + data[at + 2]) / 3;
    sum += grey;
    squares += grey * grey;
  }
  const pixels = data.length / 4;
  return { width: canvas.width, height: canvas.height, colours: colours.size, spread: Math.sqrt(squares / pixels - (sum / pixels) ** 2) };
}

/** In the page: the canvas's RGBA pixels, base64-encoded. */
function canvasPixels() {
  const canvas = document.querySelector('#picture');
  const { data } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  // Spread in slices: one call takes only so many arguments.
  const SLICE = 0x8000;
  let binary = '';
  for (let at = 0; at < data.length; at += SLICE) binary += String.fromCharCode(...data.subarray(at, at + SLICE));
  return btoa(binary);
}

/** In the page: the type and size of the file the download link holds. */
async function downloaded() {
  const blob = await (await fetch(document.querySelector('#download').href)).blob();
  return { type: blob.type, size: blob.size };
}

/** In the page: forgets the cached bundle `name`, so the profile holds a few bundles, not all. */
async function forget(name) {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(`iris-${name}`, { recursive: true });
}

/** The bytes in which two buffers differ, those past the shorter one included. */
function differing(actual, expected) {
  let count = Math.abs(actual.length - expected.length);
  for (let at = 0; at < Math.min(actual.length, expected.length); at++) count += actual[at] !== expected[at];
  return count;
}

/** Runs the single-threaded twin of the page's build in Node on `bundle` with `settings`, which writes `<bundle>.rgba` to `out`. */
function runReference(bundle, settings, out) {
  const smoke = join(ROOT, 'scripts/node-smoke.mjs');
  const build = values.build.replace('threaded', 'single');
  const options = Object.entries(settings).flatMap(([field, value]) => [`--${field}`, String(value)]);
  const args = [smoke, join(ROOT, 'dist/model'), '--build', build, '--modes', bundle, '--out', out, ...options];
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { stdio: 'inherit', cwd: ROOT });
    child.once('error', reject);
    child.once('exit', (code) => (code === 0 ? resolve() : reject(new Error(`the reference of ${bundle} exited with ${code}`))));
  });
}

/** Sets the range or text input `id` to `value`, as typing would. */
const setInput = (page, id, value) =>
  page.$eval(`#${id}`, (input, text) => {
    input.value = text;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, String(value));

const choose = (page, name, value) => page.click(`.choice:has(input[name="${name}"][value="${value}"]) span`);

/** Fills the form of `mode` and returns the settings the reference runs. */
async function fill(page, { mode, photo }) {
  if (photo) {
    await page.setInputFiles('#photo', join(ROOT, photo));
    await page.waitForFunction((file) => document.querySelector('#photo-description')?.textContent.startsWith(file), photo.split('/').at(-1));
    if (mode === 'depth') await choose(page, 'max-side', MAX_SIDE);
    if (mode === 'upscale') await choose(page, 'budget', BUDGET);
    return { 'depth-photo': join(ROOT, photo), 'upscale-photo': join(ROOT, photo), 'max-side': MAX_SIDE, budget: BUDGET };
  }
  await choose(page, 'size', SIZE.id);
  await setInput(page, 'seed', SEED);
  const settings = { prompt: await page.inputValue('#prompt'), width: SIZE.width, height: SIZE.height, steps: STEPS, seed: SEED };
  if (mode === 'generate') {
    await setInput(page, 'steps', STEPS);
    Object.assign(settings, { negative: await page.inputValue('#negative'), guidance: await page.inputValue('#guidance'), shift: await page.inputValue('#shift') });
  }
  return settings;
}

const launch = {
  root: ROOT,
  name: 'iris',
  values,
  fields: { mode: 'turbo', size: SIZE.id },
};
await run(launch, async ({ page, requests, query, base, runDir, results, check, revisit }) => {
  const t0 = Date.now();
  await page.goto(`${base}/${query}`);
  await page.waitForSelector('main[data-phase]');
  results.metrics = {};
  results.modes = {};
  if (!values.build) throw new Error('pass --build web-threaded or web-threaded-relaxed: the reference is its single-threaded twin');

  // The references are slow (one thread): they run one after another in the background while the
  // page works, and are checked once the page is done.
  let queue = Promise.resolve();
  const captured = [];
  for (const { mode, bundle, photo, figures } of CASES) {
    const shown = {};
    results.modes[mode] = shown;
    if (mode !== 'turbo') {
      await choose(page, 'mode', mode);
      await page.waitForFunction(loading, null, { timeout: DEADLINE_MS, polling: 50 });
    }
    const reference = join(runDir, `${STAMP}-reference-${bundle}.rgba`);
    const settings = await fill(page, { mode, photo });
    const out = mkdtempSync(join(tmpdir(), 'iris-reference-'));
    queue = queue.then(() => runReference(bundle, settings, out)).then(() => {
      copyFileSync(join(out, `${bundle}.rgba`), reference);
      rmSync(out, { recursive: true, force: true });
    });
    queue.catch(() => {});

    if (mode === 'turbo') shown.queuedWhileLoading = (await page.getAttribute('main', 'data-loaded')) !== 'true';
    await page.click('#generate');
    await page.waitForFunction(loadedOrBroken, null, { timeout: DEADLINE_MS, polling: 250 });
    if (mode === 'turbo') results.loadMs = Date.now() - t0;
    const requestsAtReady = requests.length;
    if (!photo) {
      await page.waitForFunction(() => Number(document.querySelector('#steps-done')?.value) > 0, null, { timeout: DEADLINE_MS, polling: 250 });
      shown.preview = await page.evaluate(inspectCanvas);
    }
    await page.waitForFunction(settled, null, { timeout: DEADLINE_MS, polling: 500 });
    shown.runtime = await page.textContent('#runtime');
    shown.status = await page.textContent('#status');
    const tiles = ['time', 'per-step', 'steps-count', 'size', 'requests'];
    shown.tiles = Object.fromEntries(await Promise.all(tiles.map(async (id) => [id, await page.textContent(`#${id}`)])));
    for (const [name, tile] of Object.entries(figures)) results.metrics[name] = shown.tiles[tile];
    shown.download = await page.evaluate(downloaded);
    shown.image = await page.evaluate(inspectCanvas);
    const pixels = Buffer.from(await page.evaluate(canvasPixels), 'base64');
    writeFileSync(join(runDir, `${STAMP}-${bundle}.rgba`), pixels);
    shown.requestsWhileGenerating = fetched(requests, requestsAtReady);
    await page.screenshot({ path: join(runDir, `${STAMP}-${bundle}.png`), fullPage: true });
    if (mode !== 'turbo') await page.evaluate(forget, bundle);
    captured.push({ mode, bundle, photo, shown, pixels, reference });
  }

  await revisit(loadedOrBroken, DEADLINE_MS);

  await queue;
  for (const { mode, photo, shown, pixels, reference } of captured) {
    shown.differingBytes = differing(pixels, readFileSync(reference));
    check(shown.status === 'Done.', `${mode} done (${shown.runtime}): ${shown.tiles.time}, ${shown.tiles['per-step']} per step, ${shown.tiles.size}`);
    if (!photo) check(shown.preview.colours > 1, `${mode}: a preview before the image (${shown.preview.width}×${shown.preview.height}, ${shown.preview.colours} colours)`);
    check(shown.image.spread > MIN_SPREAD, `${mode}: the image is a picture: spread ${shown.image.spread.toFixed(1)}, ${shown.image.colours} colours`);
    check(shown.differingBytes === 0, `${mode}: the image equals the single-threaded runtime's in Node: ${shown.differingBytes} bytes differ`);
    check(shown.download.type === 'image/png' && shown.download.size > 0, `${mode}: the download is ${shown.download.type}, ${shown.download.size} bytes`);
    check(shown.requestsWhileGenerating.length === 0 && shown.tiles.requests === '0', `${mode}: requests while generating: ${shown.requestsWhileGenerating.length} seen, page counts ${shown.tiles.requests}`);
  }
});
