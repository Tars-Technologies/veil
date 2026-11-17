import { type ReactNode } from "react";
import type { Veil } from "./client.ts";
import { type MaskOptions } from "./detect.ts";
import type { Span } from "./engine.ts";
export declare function VeilProvider({ veil, children }: {
    veil: Veil;
    children: ReactNode;
}): import("react").JSX.Element;
export declare function useVeil(): Veil;
export type MaskResult = {
    /** The masked text; undefined while pending or on error. */
    data: string | undefined;
    spans: Span[] | undefined;
    status: "pending" | "success" | "error";
    isPending: boolean;
    error: Error | null;
};
/** Masks `options.text` again each time the options change. For a live preview; to mask on send, call `useVeil().mask`. */
export declare function useMask(options: MaskOptions): MaskResult;
