export declare const SPAN_TYPES: readonly ["PERSON", "ADDRESS", "DOB", "ID", "FINANCIAL", "PHONE", "EMAIL"];
export type SpanType = (typeof SPAN_TYPES)[number];
export type Span = {
    start: number;
    end: number;
    type: SpanType;
};
export type ModelInfo = {
    name: string;
    version: string;
    format: number;
};
type Token = {
    start: number;
    end: number;
    word: string;
};
export type Range = [start: number, end: number];
export declare const CONTEXT_CHARS = 300;
export declare const MAX_CHARS = 20000;
export declare const MAX_TOKENS = 3000;
export declare const WINDOW_TOKENS = 256;
export declare const WINDOW_STRIDE = 192;
export declare function tokenize(text: string): Token[];
/**
 * Joins spans that overlap, and with `joinTouching` also spans of one type that touch. The type of the first span is kept.
 */
export declare function mergeSpans(spans: Span[], joinTouching: boolean): Span[];
export declare class Engine {
    readonly model: ModelInfo;
    private readonly buckets;
    private readonly hidden;
    private readonly width;
    private readonly types;
    private readonly bias;
    private readonly labels;
    private readonly free;
    private readonly allowed;
    private readonly chars;
    private readonly shapes;
    private readonly lists;
    private readonly probes;
    private readonly lstm;
    private readonly kernel;
    private readonly buffers;
    private readonly outW;
    private readonly outB;
    private readonly sigmoid;
    private readonly tanh;
    constructor(bytes: ArrayBuffer);
    /**
     * Spans of the given `types` in `text`, read with `context` (the text that came before) in front of it.
     * `bias` is added to the person labels before the final choice: above 0 masks more names, below 0 masks fewer.
     * Throws on a text that is too long and on non-finite numbers, so a caller never gets a silent "nothing found".
     */
    detect(text: string, context?: string, bias?: number, types?: readonly SpanType[]): Span[];
    /** True when the word is on the model's list of common words. */
    isCommonWord(word: string): boolean;
    private flags;
    /** Adds the mean of the hashed feature vectors of a word to `out`: character n-grams, and for a word with digits its shape and length. */
    private embedChars;
    private runLstm;
    private detectWindow;
}
/** Email addresses, links and file paths in `text`, in order. No name is masked inside them. */
export declare function nonNameRanges(text: string): Range[];
export {};
