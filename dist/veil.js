// Runs the model on the calling thread: servers, serverless functions, tests. In a browser, import "@tars-technologies/veil/browser".
import { createClient } from "./client.js";
import { detect } from "./detect.js";
import { Engine } from "./engine.js";
export { version } from "./client.js";
export async function localRuntime(model) {
    const engine = new Engine(typeof model === "string" ? await download(model) : model);
    return { model: engine.model, detect: (options) => detect(engine, options) };
}
async function download(url) {
    const response = await fetch(url);
    if (!response.ok)
        throw new Error(`veil: model download failed with status ${response.status}`);
    return response.arrayBuffer();
}
export const createVeil = (options) => createClient(localRuntime, options);
