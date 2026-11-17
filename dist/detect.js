import { mergeSpans, nonNameRanges } from "./engine.js";
const NAMES = ["PERSON"];
const REPLACEMENTS = {
    PERSON: "[NAME]",
    ADDRESS: "[ADDRESS]",
    DOB: "[DATE OF BIRTH]",
    ID: "[ID]",
    FINANCIAL: "[FINANCIAL]",
    PHONE: "[PHONE]",
    EMAIL: "[EMAIL]"
};
const NOT_A_NAME = new Set([
    "skip",
    "none",
    "n/a",
    "na",
    "no",
    "nope",
    "why",
    "why?",
    "test",
    "later",
    "yes",
    "ok",
    "okay",
    "hi",
    "hello",
    "hey",
    "help",
    "thanks",
    "thank you",
    "stop",
    "exit",
    "menu",
    "back",
    "nein",
    "ja",
    "non",
    "oui",
    "si",
    "sí",
    "nee",
    "?",
    "what",
    "what?",
    "idk",
    "nothing",
    "anonymous",
    "private",
    "prefer not to say",
    "hola",
    "hallo",
    "bonjour"
]);
// A name answer is a few words; a longer text goes to the model, which enforces the size limits.
const NAME_ANSWER_CHARS = 80;
const terms = (list = []) => list
    .map((term) => term.trim().toLowerCase())
    .filter((term) => term.length >= 2);
/** The whole answer to a name question, unless it is long, holds a digit, "@", a link or a path, is a refusal, or holds a common word. */
function nameAnswer(engine, text) {
    const answer = text.trim();
    const words = answer.toLowerCase().match(/\p{L}+/gu) ?? [];
    const parts = answer.split(/\s+/).filter(Boolean);
    if (text.length > NAME_ANSWER_CHARS)
        return null;
    if (parts.length < 1 || parts.length > 4 || !words.length)
        return null;
    if (/[\p{Nd}@/]/u.test(answer) ||
        NOT_A_NAME.has(answer.toLowerCase().replace(/[.!]+$/u, "")))
        return null;
    if (words.some((word) => engine.isCommonWord(word)))
        return null;
    if (nonNameRanges(text).length)
        return null;
    const start = text.length - text.trimStart().length;
    return { start, end: start + answer.length, type: "PERSON" };
}
/** Each listed term where it stands as a whole word, outside email addresses, links and paths. */
function listed(text, always) {
    if (!always.length)
        return [];
    const blocked = nonNameRanges(text);
    const out = [];
    for (const term of always) {
        const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        for (const match of text.matchAll(new RegExp(`(?<![\\p{L}\\p{M}\\p{N}])${escaped}(?![\\p{L}\\p{M}\\p{N}])`, "giu"))) {
            const start = match.index;
            const end = start + match[0].length;
            if (!blocked.some(([from, to]) => start < to && from < end))
                out.push({ start, end, type: "PERSON" });
        }
    }
    return out;
}
export function detect(engine, options) {
    const { text, context = "", threshold = 0.5, types = NAMES } = options;
    if (!(threshold > 0 && threshold < 1))
        throw new Error("veil: threshold must be between 0 and 1");
    const whole = options.expectName && types.includes("PERSON")
        ? nameAnswer(engine, text)
        : null;
    const found = whole
        ? [whole]
        : engine.detect(text, context, Math.log((1 - threshold) / threshold), types);
    return applyTermOverrides(text, found, options);
}
/** Apply exact-term rules against the complete input, including across chunks. */
export function applyTermOverrides(text, found, options) {
    const never = new Set(terms(options.never));
    return mergeSpans([
        ...found.filter(({ start, end }) => !never.has(text.slice(start, end).trim().toLowerCase())),
        ...listed(text, terms(options.always))
    ], false);
}
export function redact(text, spans, replacement) {
    let out = "";
    let cursor = 0;
    for (const { start, end, type } of spans) {
        out += text.slice(cursor, start) + (replacement ?? REPLACEMENTS[type]);
        cursor = end;
    }
    return out + text.slice(cursor);
}
