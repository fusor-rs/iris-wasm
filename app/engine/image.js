// The image model's home: one of Iris-3B's four transformers (Turbo, the base model, depth or the
// restorer) in a module Worker with its own runtime instance and memory. The page starts a worker
// per variant and ends it when another is chosen, so one transformer is in memory at a time.
//
// Load: fusor-ai's worker kit (`dist/host/worker.js`) starts the best runtime build and streams
// every part of the bundle into it; the runtime checks each part's size and BLAKE3.
//
// Run: a job's `start_…` takes its request (and the text states, or the photo), then one `step()`
// per macrotask, so a stop request is seen between steps. A generation's step is followed by the
// clean image it predicts, a restoration's step is a tile; `take_image()` returns the finished
// image.
//
// After the model is loaded nothing else is fetched: everything is entirely local.

import { engine, load, post, requestsSinceReady, serve, yieldTask } from './dist/host/worker.js';

let stoppedJob = null;

serve({
  load: ({ modelUrl, threads, build, variant }) =>
    load({
      modelUrl,
      threads,
      build,
      glueName: 'iris_runtime',
      className: 'ImageRuntime',
      cacheName: `iris-${variant}`,
      distUrl: new URL('./dist/', import.meta.url),
      onRuntime: (started) => post({ type: 'runtime', build: started.build.name, threads: started.threads }),
      onReady: (progress) => post({ type: 'ready', ...progress }),
    }),
  generate: ({ job, request, prompt, negative }) => run(job, () => engine.rt.start_generation(request, prompt, negative), true),
  depth: ({ job, request, rgba, width, height }) => run(job, () => engine.rt.start_depth(request, rgba, width, height), false),
  upscale: ({ job, request, rgba, width, height }) => run(job, () => engine.rt.start_upscale(request, rgba, width, height), false),
  stop: ({ job }) => (stoppedJob = job),
});

/** Runs the job `start` begins to its end, previewing each step when `previews`. */
async function run(job, start, previews) {
  const { rt } = engine;
  let t0 = performance.now();
  const plan = JSON.parse(start());
  const timing = { computeMs: performance.now() - t0, stepMs: 0, step: 0 };
  let stepsMs = 0;
  post({ type: 'started', job, plan, ...timing });
  for (;;) {
    await yieldTask();
    if (stoppedJob === job) {
      rt.cancel();
      return post({ type: 'done', job, stopped: true, ...timing, networkRequests: requestsSinceReady() });
    }
    t0 = performance.now();
    const progress = JSON.parse(rt.step());
    const preview = previews && !progress.done ? { width: plan.width, height: plan.height, pixels: rt.preview() } : null;
    const elapsed = performance.now() - t0;
    stepsMs += elapsed;
    timing.computeMs += elapsed;
    timing.step = progress.step;
    timing.stepMs = stepsMs / progress.step;
    if (!progress.done) {
      post({ type: 'step', job, preview, ...timing, networkRequests: requestsSinceReady() }, preview ? [preview.pixels.buffer] : []);
      continue;
    }
    const pixels = rt.take_image();
    const image = { width: plan.width, height: plan.height, pixels };
    return post({ type: 'done', job, stopped: false, image, ...timing, networkRequests: requestsSinceReady() }, [pixels.buffer]);
  }
}
