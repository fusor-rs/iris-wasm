//! The Iris-3B page. The form, the model's loading and each job's live measurements are signals
//! here; `web/app.js` runs the engine workers, reads the photos and draws the images, and reports
//! back through `engine` events.

mod app;
