import { type Engine, type Span, type SpanType } from "./engine.ts";
export type MaskOptions = {
    text: string;
    /** The text that came before, for example the question this text answers. */
    context?: string;
    /** 0 to 1, default 0.5. Lower masks more names and more words that are not names. */
    threshold?: number;
    types?: SpanType[];
    /** Exact terms to mask wherever they stand as whole words, for example a name the person gave earlier. */
    always?: string[];
    /** Exact terms to leave, for example brand, product and assistant names. */
    never?: string[];
    /** The text is the answer to a name question: a short answer is masked as a whole. */
    expectName?: boolean;
    /** Default: the kind in brackets, for example "[NAME]" or "[ADDRESS]". */
    replacement?: string;
};
export declare function detect(engine: Engine, options: MaskOptions): Span[];
/** Apply exact-term rules against the complete input, including across chunks. */
export declare function applyTermOverrides(text: string, found: Span[], options: Pick<MaskOptions, "always" | "never">): Span[];
export declare function redact(text: string, spans: Span[], replacement?: string): string;
