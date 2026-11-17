import { type MaskOptions } from "./detect.ts";
import type { ModelInfo, Span, SpanType } from "./engine.ts";
export type { MaskOptions, ModelInfo, Span, SpanType };
export declare const version = "0.3.1";
export type Defaults = Omit<MaskOptions, "text">;
export type VeilOptions = {
    /** The bytes of the model file, or a URL to it. */
    model: ArrayBuffer | string;
    /** Options applied to every call; a call can override each of them. */
    defaults?: Defaults;
};
export type Veil = {
    /** "idle" until the first call or `ready()`. */
    readonly status: "idle" | "loading" | "ready" | "error";
    /** Name, version and format of the loaded model; null until ready. */
    readonly model: ModelInfo | null;
    /** The defaults as given to `createVeil`. They cannot change; create a new client for other defaults. */
    readonly defaults: Readonly<Defaults>;
    /** Loads the model. Optional: `detect` and `mask` load it on first use. */
    ready(): Promise<void>;
    /** Spans of personal data in `text`, as UTF-16 offsets, in order, without overlap. */
    detect(options: MaskOptions): Promise<Span[]>;
    /** `text` with every span replaced. */
    mask(options: MaskOptions): Promise<string>;
};
/** A loaded model, on this thread or behind a worker. */
export type Runtime = {
    model: ModelInfo;
    detect(options: MaskOptions): Span[] | Promise<Span[]>;
};
export declare function createClient(start: (model: ArrayBuffer | string, fail: (error: Error) => void) => Promise<Runtime>, options: VeilOptions): Veil;
