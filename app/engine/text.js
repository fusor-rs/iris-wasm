// The text encoder's home: Qwen3-VL's language model in a module Worker with its own runtime
// instance and memory (a wasm32 memory holds it or a transformer, not both). It turns each prompt
// into the states the transformer reads, which the page hands to the image worker.
//
// Load: fusor-ai's worker kit (`dist/host/worker.js`) starts the best runtime build and streams
// every part of the bundle into it; the runtime checks each part's size and BLAKE3.

import { engine, load, post, requestsSinceReady, serve } from './dist/host/worker.js';

serve({
  load: ({ modelUrl, threads, build }) =>
    load({
      modelUrl,
      threads,
      build,
      glueName: 'iris_runtime',
      className: 'TextRuntime',
      cacheName: 'iris-text',
      distUrl: new URL('./dist/', import.meta.url),
      onRuntime: (started) => post({ type: 'runtime', build: started.build.name, threads: started.threads }),
      onReady: (progress) => post({ type: 'ready', ...progress }),
    }),
  // The negative prompt is null when no guidance reads it: its states are then empty.
  encode: ({ job, prompt, negative }) => {
    const t0 = performance.now();
    const states = { prompt: engine.rt.encode(prompt), negative: negative === null ? new Float32Array(0) : engine.rt.encode(negative) };
    post({ type: 'encoded', job, ...states, textMs: performance.now() - t0, networkRequests: requestsSinceReady() }, [states.prompt.buffer, states.negative.buffer]);
  },
});
