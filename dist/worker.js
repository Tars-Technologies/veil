import { localRuntime } from "./veil.js";
let runtime;
self.onmessage = async ({ data }) => {
    try {
        if (data.model !== undefined)
            runtime = localRuntime(data.model);
        if (!runtime)
            throw new Error("veil: the worker has no model");
        const loaded = await runtime;
        const reply = {
            id: data.id,
            result: data.options ? await loaded.detect(data.options) : loaded.model
        };
        self.postMessage(reply);
    }
    catch (error) {
        const reply = {
            id: data.id,
            error: error instanceof Error ? error.message : String(error)
        };
        self.postMessage(reply);
    }
};
