// The same API as "@tars-technologies/veil", with the model in a Web Worker, so the page never waits for it.
import { createClient } from "./client.js";
export { version } from "./client.js";
async function workerRuntime(model, fail) {
    const worker = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module"
    });
    const waiting = new Map();
    let next = 0;
    worker.onmessage = ({ data }) => {
        const entry = waiting.get(data.id);
        waiting.delete(data.id);
        if (data.error === undefined)
            entry?.resolve(data.result);
        else
            entry?.reject(new Error(data.error));
    };
    // A worker that fails to load or throws outside a request would leave every call waiting, now and later.
    worker.onerror = (event) => {
        const error = new Error(`veil: the worker failed: ${event.message}`);
        for (const entry of waiting.values())
            entry.reject(error);
        waiting.clear();
        worker.terminate();
        fail(error);
    };
    const call = (message) => new Promise((resolve, reject) => {
        waiting.set(next, {
            resolve: resolve,
            reject
        });
        const request = { id: next++, ...message };
        worker.postMessage(request);
    });
    // A relative URL must resolve against the page; inside the worker it would resolve against the worker script.
    const source = typeof model === "string" ? new URL(model, document.baseURI).href : model;
    return {
        model: await call({ model: source }),
        detect: (options) => call({ options })
    };
}
export const createVeil = (options) => createClient(workerRuntime, options);
