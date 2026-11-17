import type { MaskOptions } from "./detect.ts";
export type Request = {
    id: number;
    model?: ArrayBuffer | string;
    options?: MaskOptions;
};
export type Reply = {
    id: number;
    result?: unknown;
    error?: string;
};
