import { redact } from "./detect.js";
export const version = "0.3.1";
export function createClient(start, options) {
    const { model } = options;
    // A copy that cannot change: `mask`, `detect` and the React hook then always agree on the defaults.
    const defaults = Object.freeze({
        ...options.defaults,
        always: Object.freeze([...(options.defaults?.always ?? [])]),
        never: Object.freeze([...(options.defaults?.never ?? [])]),
        types: options.defaults?.types &&
            Object.freeze([...options.defaults.types])
    });
    let status = "idle";
    let info = null;
    let runtime = null;
    // A runtime that breaks after loading: every later call rejects with its error.
    const fail = (error) => {
        status = "error";
        runtime = Promise.reject(error);
        runtime.catch(() => { });
    };
    const load = () => {
        if (!runtime) {
            status = "loading";
            runtime = start(model, fail).then((loaded) => {
                status = "ready";
                info = loaded.model;
                return loaded;
            }, (error) => {
                status = "error";
                throw error;
            });
        }
        return runtime;
    };
    // The options are resolved and copied before the first wait, so a caller that changes its object or its lists
    // afterwards cannot change the result. A default applies when the call gives no value.
    const run = async (options) => {
        const types = options.types ?? defaults.types;
        const call = {
            text: options.text,
            context: options.context ?? defaults.context,
            threshold: options.threshold ?? defaults.threshold,
            types: types && [...types],
            always: [...(options.always ?? defaults.always ?? [])],
            never: [...(options.never ?? defaults.never ?? [])],
            expectName: options.expectName ?? defaults.expectName,
            replacement: options.replacement ?? defaults.replacement
        };
        const spans = await (await load()).detect(call);
        return { call, spans };
    };
    return {
        get status() {
            return status;
        },
        get model() {
            return info;
        },
        defaults,
        ready: async () => void (await load()),
        detect: async (options) => (await run(options)).spans,
        mask: async (options) => {
            const { call, spans } = await run(options);
            return redact(call.text, spans, call.replacement);
        }
    };
}
