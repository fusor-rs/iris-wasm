import { ComputePool, SPAWN_COMPUTE } from '../../vendor/fusor-ai/packages/host/compute.js';

// The engine bridge. The page's state lives in Rust (src/app.rs); this module owns the engine
// workers (the text encoder's, and the image model's of the chosen mode, each with its own memory)
// and, in threaded builds, their compute workers (fusor-ai's ComputePool). A prompt is encoded in
// the text worker and its states handed to the image worker, which draws each step's preview on
// the canvas; a photo is read here and handed to the image worker as pixels. It measures each job
// and reports back with `engine` events.

const params = new URLSearchParams(location.search);
const cores = navigator.hardwareConcurrency || 4;
// One compute worker per core minus two (the page and the engine worker), at most 8; `?threads=1`
// forces the single-threaded build. Never more than the cores: the barriers spin.
const requestedThreads = Number.parseInt(params.get('threads') ?? '', 10);
const THREADS = Math.min(cores, requestedThreads > 0 ? requestedThreads : Math.max(1, Math.min(8, cores - 2)));
// The bundles served next to the page (`just serve` mounts dist/model there); `?model=<url>` loads
// another one, `?build=web-single` pins a runtime build.
// A Hugging Face Space holds at most 1 GB, so the Space loads the bundles from the model repo
// fusorlabs/iris-3b-web (`just deploy-model`) instead of from beside the page.
const HOSTED_MODEL = 'https://huggingface.co/fusorlabs/iris-3b-web/resolve/main/';
const DEFAULT_MODEL = location.hostname.endsWith('.hf.space') ? HOSTED_MODEL : 'model/';
const MODEL_URL = new URL(params.get('model') ?? DEFAULT_MODEL, document.baseURI).href;
// The transformer each mode runs; the modes that read a prompt also need the text encoder.
const VARIANTS = { turbo: 'turbo', generate: 'base', depth: 'depth', upscale: 'upscaler' };
const PROMPT_MODES = new Set(['turbo', 'generate']);

const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;

export function onMount({ root, inputs, onCleanup }) {
  const report = (detail) => root.dispatchEvent(new CustomEvent('engine', { detail: JSON.stringify(detail) }));
  let notice = '';
  let loaded = false;
  let broken = false;
  let jobs = 0;
  let job = null;
  let photo = null;
  let imageUrl = null;
  let readyAt = Infinity;
  let mode = null;
  const bundles = { text: null, image: null };

  const handlers = {
    text: {
      encoded: (message) => {
        if (message.job !== job?.id || job.stopped) return;
        job.textMs = message.textMs;
        job.textRequests = message.networkRequests;
        job.encoding = false;
        report({ type: 'progress', status: 'Denoising…', metrics: metrics() });
        const { prompt, negative } = message;
        bundles.image.worker.postMessage({ type: 'generate', job: job.id, request: inputs.request.get(), prompt, negative }, [prompt.buffer, negative.buffer]);
      },
    },
    image: {
      started: (message) => {
        if (message.job !== job?.id) return;
        job.plan = message.plan;
        job.engine = message;
        drawPhoto();
      },
      step: (message) => {
        if (message.job !== job?.id) return;
        job.engine = message;
        if (message.preview) draw(message.preview, '#picture');
        report({ type: 'progress', status: stepStatus(message.step), metrics: metrics() });
      },
      done: (message) => finish(message),
    },
  };

  /** What the job's step `step` just finished, for the status line. */
  function stepStatus(step) {
    const { steps } = job.plan;
    if (job.mode === 'upscale') return step < steps ? `Restoring tile ${step + 1} of ${steps}…` : 'Done.';
    return `Step ${step} of ${steps}`;
  }

  /** Starts the worker of the bundle `directory` as the text encoder's or the image model's. */
  function start(name, directory) {
    const worker = new Worker(new URL(`engine/${name}.js`, document.baseURI), { type: 'module', name: `iris-${directory}` });
    const pool = new ComputePool(crash);
    const bundle = { name, directory, worker, pool, build: null, progress: { done: 0, total: 0, network: 0, cache: 0 }, ready: null };
    const routes = {
      [SPAWN_COMPUTE]: (message) => pool.spawn(message),
      runtime: (message) => (bundle.build = message),
      warning: ({ message }) => {
        console.warn(message);
        notice = ` (${message})`;
      },
      progress: (message) => {
        bundle.progress = message;
        reportLoading();
      },
      ready: (message) => {
        bundle.ready = message;
        if (!loaded && needed().every((each) => each.ready)) ready();
      },
      error: ({ message }) => {
        if (!loaded) return fail(message);
        if (job) job.encoding = false;
        report({ type: 'error', message });
      },
      fatal: ({ message }) => fail(message),
      ...handlers[name],
    };
    worker.onmessage = ({ data }) => {
      if (!broken) routes[data.type]?.(data);
    };
    worker.onerror = (event) => crash(event.message || 'an engine worker failed');
    worker.postMessage({ type: 'load', modelUrl: new URL(`${directory}/`, MODEL_URL).href, threads: THREADS, build: params.get('build'), variant: directory });
    bundles[name] = bundle;
  }

  function stopWorker(name) {
    bundles[name]?.worker.terminate();
    bundles[name]?.pool.terminate();
    bundles[name] = null;
  }

  /** The bundles the mode on show runs. */
  const needed = () => [bundles.image, PROMPT_MODES.has(mode) ? bundles.text : null].filter(Boolean);

  const total = (key) => needed().reduce((sum, bundle) => sum + bundle.progress[key], 0);

  function reportLoading() {
    if (loaded) return;
    const [done, size, network, cache] = ['done', 'total', 'network', 'cache'].map(total);
    const source = cache && !network ? 'from this device' : cache ? `${mb(network)} downloaded, ${mb(cache)} cached` : 'downloading';
    report({ type: 'loading', fraction: size ? done / size : 0, detail: `${mb(done)} / ${mb(size)} · ${source}` });
  }

  /** Loads what `next` needs and lets go of an image model another mode loaded. */
  function select(next) {
    if (next === mode) return;
    mode = next;
    loaded = false;
    if (bundles.image?.directory !== VARIANTS[mode]) {
      stopWorker('image');
      start('image', VARIANTS[mode]);
    }
    if (PROMPT_MODES.has(mode) && !bundles.text) start('text', 'text');
    // A subscription callback runs inside Rust's update: report once it has returned.
    queueMicrotask(() => (needed().every((each) => each.ready) ? ready() : reportLoading()));
  }

  function ready() {
    loaded = true;
    readyAt = performance.now();
    const parts = needed();
    const bundleBytes = parts.reduce((sum, bundle) => sum + bundle.ready.total, 0);
    const cached = parts.reduce((sum, bundle) => sum + bundle.ready.cache, 0);
    const loadedInfo = {
      build: `${bundles.image.build.build}${notice}`,
      threads: bundles.image.build.threads,
      bundle_bytes: bundleBytes,
      load_ms: Math.max(...parts.map((bundle) => bundle.ready.ms)),
      from_device: cached === bundleBytes,
    };
    report({ type: 'ready', loaded: loadedInfo });
  }

  /** The photo chosen, as RGBA the way the reference reads it: no colour conversion, no premultiplied alpha. */
  async function readPhoto(file) {
    const bitmap = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d');
    context.drawImage(bitmap, 0, 0);
    const { data, width, height } = context.getImageData(0, 0, bitmap.width, bitmap.height);
    photo = { width, height, rgba: new Uint8Array(data.buffer) };
    report({ type: 'photo', description: `${file.name} · ${width}×${height}` });
  }

  /** Runs the request the page holds, timed from now (the click, or the model being ready). */
  function dispatch() {
    job = { id: ++jobs, mode, t0: performance.now(), encoding: PROMPT_MODES.has(mode), stopped: false, plan: null, engine: null, textMs: 0, textRequests: 0 };
    if (job.encoding) {
      bundles.text.worker.postMessage({ type: 'encode', job: job.id, ...JSON.parse(inputs.prompts.get()) });
      return;
    }
    const rgba = photo.rgba.slice();
    const message = { type: mode, job: job.id, request: inputs.request.get(), rgba, width: photo.width, height: photo.height };
    bundles.image.worker.postMessage(message, [rgba.buffer]);
  }

  function stop() {
    if (!job) return;
    job.stopped = true;
    if (job.encoding) {
      job.encoding = false;
      // A subscription callback runs inside Rust's update: report once it has returned.
      queueMicrotask(() => report({ type: 'done', status: 'Stopped.', image: null, metrics: metrics() }));
      return;
    }
    bundles.image.worker.postMessage({ type: 'stop', job: job.id });
  }

  /** Puts RGBA `{ width, height, pixels }` on the canvas `selector`, which takes the image's size. */
  function draw({ width, height, pixels }, selector) {
    const canvas = root.querySelector(selector);
    if (!canvas) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const image = new ImageData(new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.length), width, height);
    canvas.getContext('2d').putImageData(image, 0, 0);
  }

  /** Shows the photo of a photo mode beside the image. */
  function drawPhoto() {
    if (photo && !PROMPT_MODES.has(job.mode)) draw({ width: photo.width, height: photo.height, pixels: photo.rgba }, '#source');
  }

  function metrics() {
    const engineStats = job.engine ?? {};
    const ownRequests = performance.getEntriesByType('resource').filter((entry) => entry.startTime >= readyAt).length;
    return {
      wall_ms: performance.now() - job.t0,
      text_ms: job.textMs,
      compute_ms: engineStats.computeMs ?? 0,
      step_ms: engineStats.stepMs ?? 0,
      step: engineStats.step ?? 0,
      steps: job.plan?.steps ?? 0,
      width: job.plan?.width ?? 0,
      height: job.plan?.height ?? 0,
      requests: ownRequests + job.textRequests + (engineStats.networkRequests ?? 0),
    };
  }

  async function finish(message) {
    if (message.job !== job?.id) return;
    job.engine = message;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = null;
    if (!message.stopped) {
      draw(message.image, '#picture');
      imageUrl = URL.createObjectURL(await new Promise((resolve) => root.querySelector('#picture').toBlob(resolve, 'image/png')));
    }
    report({ type: 'done', status: message.stopped ? 'Stopped.' : 'Done.', image: imageUrl, metrics: metrics() });
  }

  function terminate() {
    for (const name of Object.keys(bundles)) stopWorker(name);
  }

  /** The model could not load, or an engine or a compute thread died. */
  function fail(message) {
    broken = true;
    terminate();
    report({ type: 'broken', message });
  }

  function crash(message) {
    fail(`The engine stopped: ${message}.`);
  }

  root.addEventListener('change', (event) => {
    const [file] = event.target.id === 'photo' ? event.target.files : [];
    if (file) readPhoto(file).catch((error) => report({ type: 'error', message: `The photo could not be read: ${error.message}` }));
  });
  select(inputs.mode.get());
  onCleanup(inputs.mode.subscribe(select));
  onCleanup(onClicks(inputs.starts, dispatch));
  onCleanup(onClicks(inputs.stops, stop));
  onCleanup(() => {
    terminate();
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  });
}

/** Runs `action` once per click a click counter counts, clicks before this module loaded included. */
function onClicks(counter, action) {
  let seen = 0;
  return counter.subscribe((count) => {
    for (; seen < count; seen++) action();
  });
}
