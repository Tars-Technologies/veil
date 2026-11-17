import { type Runtime, type Veil, type VeilOptions } from "./client.ts";
export { type Defaults, type MaskOptions, type ModelInfo, type Span, type SpanType, type Veil, type VeilOptions, version } from "./client.ts";
export declare function localRuntime(model: ArrayBuffer | string): Promise<Runtime>;
export declare const createVeil: (options: VeilOptions) => Veil;
