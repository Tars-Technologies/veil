import type { Veil } from "./client.ts";
import { type MaskOptions } from "./detect.ts";
import { type Span } from "./engine.ts";
export type DetectionOptions = Omit<MaskOptions, "text" | "replacement">;
/** Detect with options in bounded, overlapping calls to one client.
 * Keeps the model's total character bound; the application may enforce less.
 * Returns only after every chunk succeeds. All offsets refer to the input.
 */
export declare function detectChunked(veil: Pick<Veil, "detect"> & Partial<Pick<Veil, "defaults">>, text: string, options?: DetectionOptions): Promise<Span[]>;
