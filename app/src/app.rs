use fusor::{JsInputs, Signal, signal};
use serde::{Deserialize, Serialize};

const SAMPLE_PROMPT: &str = "A red fox sleeping in fresh snow, golden hour";
const DEFAULT_STEPS: f64 = 100.0;
const DEFAULT_GUIDANCE: f64 = 3.0;
const DEFAULT_SHIFT: f64 = 4.0;
const DEFAULT_SEED: &str = "42";
const DEFAULT_SIZE: &str = "1024x1024";
const DEFAULT_MAX_SIDE: &str = "1024";
const DEFAULT_BUDGET: &str = "256";
// Turbo takes its four levels from its distillation: only the size and the seed are the page's.
const TURBO_STEPS: usize = 4;
const TURBO_GUIDANCE: f32 = 1.0;
const TURBO_SHIFT: f32 = 4.0;

/// The sizes Iris-3B was trained to generate (`iris::model::SIZES`): id, label, width, height.
const SIZES: [(&str, &str, u32, u32); 7] = [
    ("1024x1024", "Square", 1024, 1024),
    ("896x1200", "Portrait 3:4", 896, 1200),
    ("1200x896", "Landscape 4:3", 1200, 896),
    ("896x1344", "Portrait 2:3", 896, 1344),
    ("1344x896", "Landscape 3:2", 1344, 896),
    ("752x1344", "Portrait 9:16", 752, 1344),
    ("1344x752", "Landscape 16:9", 1344, 752),
];
/// One pass of the transformer over 1024 × 1024 pixels, as the e2e run measured it with relaxed
/// SIMD on an 8-thread Apple M4 Pro: the base of the time estimates, which scale with the passes
/// and the pixels.
const PASS_SECONDS: f64 = 18.6;
const PASS_PIXELS: f64 = 1024.0 * 1024.0;

/// The longest side the depth model sees: id (0 keeps the photo's), label.
const MAX_SIDES: [(&str, &str); 3] = [("512", "512 px"), ("1024", "1024 px"), ("0", "The photo's")];

/// The sides a restored photo is shrunk to before it is enlarged four times: id, label.
const BUDGETS: [(&str, &str); 3] = [("256", "256 px"), ("384", "384 px"), ("512", "512 px")];

/// The modes: id, label, what they do.
const MODES: [(&str, &str, &str); 4] = [
    ("turbo", "Turbo", "Four steps"),
    ("generate", "Generate", "The base model"),
    ("depth", "Depth", "A photo's depth"),
    ("upscale", "Upscale", "Four times larger"),
];

#[derive(Clone, PartialEq)]
struct Choice {
    id: String,
    label: String,
    detail: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Turbo,
    Generate,
    Depth,
    Upscale,
}

impl Mode {
    fn parse(id: &str) -> Self {
        match id {
            "turbo" => Self::Turbo,
            "generate" => Self::Generate,
            "depth" => Self::Depth,
            "upscale" => Self::Upscale,
            other => unreachable!("the mode radio's values are the ids of MODES, not {other}"),
        }
    }

    fn reads_prompt(self) -> bool {
        matches!(self, Self::Turbo | Self::Generate)
    }

    fn reads_photo(self) -> bool {
        !self.reads_prompt()
    }
}

// The runtime build and the bundles, once loaded.
#[derive(Clone, PartialEq, Deserialize)]
struct Loaded {
    build: String,
    threads: u32,
    bundle_bytes: f64,
    load_ms: f64,
    // Every part of every bundle the mode needs came from this device's storage.
    from_device: bool,
}

impl Loaded {
    fn summary(&self) -> String {
        let threads = match self.threads {
            1 => "1 thread".to_string(),
            count => format!("{count} threads"),
        };
        let source = if self.from_device {
            "model loaded from this device"
        } else {
            "model downloaded"
        };
        format!(
            "{} · {threads} · {:.0} MB bundles · {source} in {:.1} s",
            self.build,
            self.bundle_bytes / 1e6,
            self.load_ms / 1000.0
        )
    }
}

// One job as `web/app.js` measures it, live while it runs: `wall_ms` on the page from the click
// (for a queued request, from the model being ready), `text_ms` (the text encoder), `compute_ms`
// (time inside the image runtime's `start_…` and `step`, summed) and `step_ms` (the mean of its
// steps) inside the engine workers, and the network requests made since the model loaded.
#[derive(Clone, Default, PartialEq, Deserialize)]
struct Metrics {
    wall_ms: f64,
    text_ms: f64,
    compute_ms: f64,
    step_ms: f64,
    step: u32,
    steps: u32,
    width: u32,
    height: u32,
    requests: u32,
}

impl Metrics {
    fn time(&self) -> String {
        format!("{:.1} s", self.wall_ms / 1000.0)
    }

    fn compute(&self) -> String {
        format!(
            "{:.1} s computing, prompt {:.1} s",
            self.compute_ms / 1000.0,
            self.text_ms / 1000.0
        )
    }

    fn per_step(&self) -> String {
        if self.step == 0 {
            return "–".into();
        }
        format!("{:.2} s", self.step_ms / 1000.0)
    }

    fn steps(&self) -> String {
        format!("{} / {}", self.step, self.steps)
    }

    fn size(&self) -> String {
        if self.width == 0 {
            return "–".into();
        }
        format!("{}×{}", self.width, self.height)
    }

    fn requests(&self) -> String {
        self.requests.to_string()
    }
}

// What `web/app.js` reports, one JSON object per `engine` event.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Report {
    // A bundle of the mode is downloading or loading from this device.
    Loading {
        fraction: f64,
        detail: String,
    },
    Ready {
        loaded: Loaded,
    },
    // The photo of a photo mode: its name and size.
    Photo {
        description: String,
    },
    Progress {
        status: String,
        metrics: Metrics,
    },
    // The image is finished; `image` is the PNG's blob URL.
    Done {
        status: String,
        image: Option<String>,
        metrics: Metrics,
    },
    // A job failed; the engine can run the next one.
    Error {
        message: String,
    },
    // The model could not load, or the engine died: nothing more can run in this page.
    Broken {
        message: String,
    },
}

// The image engine's generation request (`iris::session::Generation` in the runtime), as
// `web/app.js` posts it.
#[derive(Serialize)]
struct Generation {
    width: u32,
    height: u32,
    steps: usize,
    guidance: f32,
    shift: f32,
    seed: u64,
}

// `iris::session::DepthRequest`.
#[derive(Serialize)]
struct DepthRequest {
    max_side: usize,
}

// `iris::session::Upscale`: the photo is shrunk to fit these sides first.
#[derive(Serialize)]
struct Upscale {
    short_side: usize,
    long_side: usize,
}

// What the text encoder reads, as `web/app.js` posts it: `negative` is null when no guidance
// reads it.
#[derive(Serialize)]
struct Prompts {
    prompt: String,
    negative: Option<String>,
}

// One job as JSON: the request and the prompts.
struct Job {
    request: String,
    prompts: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Loading,
    Ready,
    Generating,
    Broken,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Generating => "generating",
            Self::Broken => "broken",
        }
    }
}

#[derive(JsInputs)]
struct App {
    // `web/app.js` loads the bundles the mode needs whenever it changes.
    #[js]
    mode: Signal<String>,
    prompt: Signal<String>,
    negative: Signal<String>,
    size: Signal<String>,
    steps: Signal<f64>,
    guidance: Signal<f64>,
    shift: Signal<f64>,
    seed: Signal<String>,
    max_side: Signal<String>,
    budget: Signal<String>,
    // `web/app.js` acts on every new count: `starts` runs `request` on `prompts`, `stops` stops
    // the job.
    #[js]
    request: Signal<String>,
    #[js]
    prompts: Signal<String>,
    #[js]
    starts: Signal<u32>,
    #[js]
    stops: Signal<u32>,
    phase: Signal<Phase>,
    // Generate was clicked while the model loaded: it runs as soon as the model is ready.
    queued: Signal<bool>,
    fraction: Signal<f64>,
    detail: Signal<String>,
    status: Signal<String>,
    image: Signal<Option<String>>,
    // The photo chosen for a photo mode.
    photo: Signal<Option<String>>,
    metrics: Signal<Metrics>,
    measured: Signal<bool>,
    loaded: Signal<Option<Loaded>>,
}

impl App {
    fn new() -> Self {
        Self {
            mode: signal("turbo".into()),
            prompt: signal(SAMPLE_PROMPT.into()),
            negative: signal(String::new()),
            size: signal(DEFAULT_SIZE.into()),
            steps: signal(DEFAULT_STEPS),
            guidance: signal(DEFAULT_GUIDANCE),
            shift: signal(DEFAULT_SHIFT),
            seed: signal(DEFAULT_SEED.into()),
            max_side: signal(DEFAULT_MAX_SIDE.into()),
            budget: signal(DEFAULT_BUDGET.into()),
            request: signal(String::new()),
            prompts: signal(String::new()),
            starts: signal(0),
            stops: signal(0),
            phase: signal(Phase::Loading),
            queued: signal(false),
            fraction: signal(0.0),
            detail: signal("Starting…".into()),
            status: signal(String::new()),
            image: signal(None),
            photo: signal(None),
            metrics: signal(Metrics::default()),
            measured: signal(false),
            loaded: signal(None),
        }
    }

    fn kind(&self) -> Mode {
        Mode::parse(&self.mode.get())
    }

    fn modes(&self) -> Vec<Choice> {
        choices(
            MODES
                .iter()
                .map(|&(id, label, detail)| (id, label, detail.into())),
        )
    }

    fn sizes(&self) -> Vec<Choice> {
        choices(
            SIZES
                .iter()
                .map(|&(id, label, width, height)| (id, label, format!("{width}×{height}"))),
        )
    }

    fn max_sides(&self) -> Vec<Choice> {
        choices(
            MAX_SIDES
                .iter()
                .map(|&(id, label)| (id, label, String::new())),
        )
    }

    fn budgets(&self) -> Vec<Choice> {
        choices(
            BUDGETS
                .iter()
                .map(|&(id, label)| (id, label, String::new())),
        )
    }

    fn generate(&self) {
        let Some(job) = self.job() else {
            return;
        };
        if self.phase.get() == Phase::Loading {
            self.queued.set(true);
            self.status
                .set("Queued: it starts the moment the model is ready.".into());
        } else {
            self.begin(job);
        }
    }

    // The job for the form's settings, or `None` after telling what is wrong.
    fn job(&self) -> Option<Job> {
        let mode = self.kind();
        if mode.reads_photo() && self.photo.get().is_none() {
            self.status.set("Choose a photo first.".into());
            return None;
        }
        let (request, prompts) = match mode {
            Mode::Turbo | Mode::Generate => (self.generation()?, self.prompts()?),
            Mode::Depth => (
                json(&DepthRequest {
                    max_side: parse(&self.max_side.get()),
                }),
                String::new(),
            ),
            Mode::Upscale => {
                let budget = parse(&self.budget.get());
                (
                    json(&Upscale {
                        short_side: budget,
                        long_side: budget,
                    }),
                    String::new(),
                )
            }
        };
        Some(Job { request, prompts })
    }

    // The generation request as JSON, or `None` after telling what is wrong with the seed.
    fn generation(&self) -> Option<String> {
        let Ok(seed) = self.seed.get().trim().parse::<u64>() else {
            self.status
                .set("The seed is a whole number from 0 to 18446744073709551615.".into());
            return None;
        };
        let (width, height) = self.dimensions();
        let request = if self.kind() == Mode::Turbo {
            Generation {
                width,
                height,
                steps: TURBO_STEPS,
                guidance: TURBO_GUIDANCE,
                shift: TURBO_SHIFT,
                seed,
            }
        } else {
            Generation {
                width,
                height,
                steps: self.steps.get() as usize,
                guidance: self.guidance.get() as f32,
                shift: self.shift.get() as f32,
                seed,
            }
        };
        Some(json(&request))
    }

    // The prompts as JSON, or `None` after telling the prompt is empty.
    fn prompts(&self) -> Option<String> {
        let prompt = self.prompt.get().trim().to_string();
        if prompt.is_empty() {
            self.status.set("Describe the image first.".into());
            return None;
        }
        let negative = self.negative.get().trim().to_string();
        let guided = self.kind() == Mode::Generate && self.guidance.get() > 1.0;
        let negative = guided.then_some(negative);
        Some(json(&Prompts { prompt, negative }))
    }

    fn begin(&self, job: Job) {
        self.request.set(job.request);
        self.prompts.set(job.prompts);
        self.phase.set(Phase::Generating);
        self.measured.set(true);
        self.image.set(None);
        self.metrics.set(Metrics::default());
        self.status.set(self.starting_status().into());
        self.starts.update(|count| *count += 1);
    }

    fn starting_status(&self) -> &'static str {
        match self.kind() {
            Mode::Turbo | Mode::Generate => "Reading the prompt…",
            Mode::Depth => "Estimating depth…",
            Mode::Upscale => "Restoring tile 1…",
        }
    }

    fn stop(&self) {
        if self.queued.get() {
            self.queued.set(false);
            self.status.set("Cancelled.".into());
            return;
        }
        self.stops.update(|count| *count += 1);
    }

    fn receive(&self, event: web_sys::Event) {
        let json = fusor::js::event_detail::<String>(&event).expect("web/app.js sends strings");
        let report = serde_json::from_str(&json).expect("web/app.js sends engine reports");
        match report {
            Report::Loading { fraction, detail } => {
                self.loaded.set(None);
                self.phase.set(Phase::Loading);
                self.fraction.set(fraction);
                self.detail.set(detail);
            }
            Report::Ready { loaded } => self.ready(loaded),
            Report::Photo { description } => self.photo.set(Some(description)),
            Report::Progress { status, metrics } => {
                self.status.set(status);
                self.metrics.set(metrics);
            }
            Report::Done {
                status,
                image,
                metrics,
            } => {
                self.status.set(status);
                self.image.set(image);
                self.metrics.set(metrics);
                self.phase.set(Phase::Ready);
            }
            Report::Error { message } => {
                self.status.set(format!("error: {message}"));
                self.phase.set(Phase::Ready);
            }
            Report::Broken { message } => {
                self.detail.set(message);
                self.phase.set(Phase::Broken);
            }
        }
    }

    fn ready(&self, loaded: Loaded) {
        self.loaded.set(Some(loaded));
        self.phase.set(Phase::Ready);
        let queued = self.queued.get();
        self.queued.set(false);
        if queued && let Some(job) = self.job() {
            self.begin(job);
        }
    }

    fn summary(&self) -> String {
        self.loaded
            .get()
            .map_or(String::new(), |loaded| loaded.summary())
    }

    fn generate_label(&self) -> &'static str {
        match (self.phase.get(), self.queued.get()) {
            (Phase::Loading, true) => "Queued",
            (Phase::Loading, false) => "Run when ready",
            _ => self.run_label(),
        }
    }

    fn run_label(&self) -> &'static str {
        match self.kind() {
            Mode::Turbo | Mode::Generate => "Generate",
            Mode::Depth => "Estimate depth",
            Mode::Upscale => "Upscale",
        }
    }

    fn can_stop(&self) -> bool {
        self.phase.get() == Phase::Generating || self.queued.get()
    }

    // The mode cannot change while a job runs.
    fn can_choose_mode(&self) -> bool {
        self.phase.get() != Phase::Generating
    }

    // The model never loaded: likely a bundle that does not match the runtime.
    fn failed_to_load(&self) -> bool {
        self.loaded.get().is_none()
    }

    fn can_generate(&self) -> bool {
        match self.phase.get() {
            Phase::Loading => !self.queued.get(),
            Phase::Ready => true,
            Phase::Generating | Phase::Broken => false,
        }
    }

    fn reads_prompt(&self) -> bool {
        self.kind().reads_prompt()
    }

    fn reads_photo(&self) -> bool {
        self.kind().reads_photo()
    }

    fn is_generate(&self) -> bool {
        self.kind() == Mode::Generate
    }

    fn is_depth(&self) -> bool {
        self.kind() == Mode::Depth
    }

    fn is_upscale(&self) -> bool {
        self.kind() == Mode::Upscale
    }

    fn photo_description(&self) -> String {
        self.photo.get().unwrap_or_else(|| "No photo chosen".into())
    }

    // What the steps tile counts.
    fn steps_note(&self) -> String {
        match self.kind() {
            Mode::Turbo => "Turbo, 4 levels".into(),
            Mode::Generate => format!("DPM++ 2M, shift {:.1}", self.shift.get()),
            Mode::Depth => "the depth pass".into(),
            Mode::Upscale => "tiles of 1024×1024".into(),
        }
    }

    // The generation's expected time at the pace the e2e run measured: Turbo's four passes, or
    // the base model's steps, each two passes when guided.
    fn estimate(&self) -> String {
        let passes = match self.kind() {
            Mode::Generate if self.guidance.get() > 1.0 => 2.0 * self.steps.get(),
            Mode::Generate => self.steps.get(),
            _ => TURBO_STEPS as f64,
        };
        let (width, height) = self.dimensions();
        let pixels = f64::from(width) * f64::from(height);
        let seconds = passes * PASS_SECONDS * pixels / PASS_PIXELS;
        let time = if seconds < 90.0 {
            format!("{seconds:.0} s")
        } else {
            format!("{:.0} min", seconds / 60.0)
        };
        let advice = if self.is_generate() {
            " Fewer steps are faster but lose detail."
        } else {
            ""
        };
        format!("About {time} on an 8-thread laptop with relaxed SIMD.{advice}")
    }

    // What the per-step tile times.
    fn step_unit(&self) -> &'static str {
        match self.kind() {
            Mode::Turbo | Mode::Generate => "per step",
            Mode::Depth => "the pass",
            Mode::Upscale => "per tile",
        }
    }

    fn image_name(&self) -> String {
        format!("iris-{}.png", self.mode.get())
    }

    fn dimensions(&self) -> (u32, u32) {
        let size = self.size.get();
        let &(_, _, width, height) = SIZES
            .iter()
            .find(|(id, ..)| *id == size)
            .expect("the size radio's values are the ids of SIZES");
        (width, height)
    }

    // The picture's aspect ratio: the job's once planned, else the chosen size's, so the frame
    // keeps its shape before the first preview.
    fn aspect(&self) -> String {
        let metrics = self.metrics.get();
        let (width, height) = if metrics.width > 0 {
            (metrics.width, metrics.height)
        } else if self.reads_prompt() {
            self.dimensions()
        } else {
            (4, 3)
        };
        format!("aspect-ratio: {width} / {height}")
    }
}

fn choices<'a>(items: impl Iterator<Item = (&'a str, &'a str, String)>) -> Vec<Choice> {
    items
        .map(|(id, label, detail)| Choice {
            id: id.into(),
            label: label.into(),
            detail,
        })
        .collect()
}

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("plain data serializes")
}

fn parse(id: &str) -> usize {
    id.parse()
        .expect("the radio's values are the ids of its table")
}

fusor::template!("web/index.html");
