import type { Veil } from "./client.ts"
import { applyTermOverrides, type MaskOptions } from "./detect.ts"
import {
  CONTEXT_CHARS,
  MAX_CHARS,
  MAX_TOKENS,
  mergeSpans,
  nonNameRanges,
  type Span,
  tokenize,
  WINDOW_STRIDE,
  WINDOW_TOKENS
} from "./engine.ts"

// Twelve model windows per call, with the same 64-token overlap as the engine.
// Leave room for the engine's bounded question context in every call.
const CHUNK_STRIDE = WINDOW_STRIDE * 12
const CHUNK_TOKENS = CHUNK_STRIDE + WINDOW_TOKENS - WINDOW_STRIDE

export type DetectionOptions = Omit<MaskOptions, "text" | "replacement">

/** Detect with options in bounded, overlapping calls to one client.
 * Keeps the model's total character bound; the application may enforce less.
 * Returns only after every chunk succeeds. All offsets refer to the input.
 */
export async function detectChunked(
  veil: Pick<Veil, "detect"> & Partial<Pick<Veil, "defaults">>,
  text: string,
  options: DetectionOptions = {}
): Promise<Span[]> {
  const resolved: DetectionOptions = { ...veil.defaults }
  for (const [key, value] of Object.entries(options))
    if (value !== undefined) Reflect.set(resolved, key, value)
  let context = (resolved.context ?? "").slice(-CONTEXT_CHARS)
  if (/^[\uDC00-\uDFFF]/u.test(context)) context = context.slice(1)
  const prefix = context ? `${context}\n` : ""
  if (text.length + prefix.length > MAX_CHARS)
    throw new Error(`veil: text longer than ${MAX_CHARS} characters`)
  if (!text) return []
  const tokens = tokenize(text)
  if (tokenize(prefix + text).length <= MAX_TOKENS)
    return veil.detect({ ...resolved, text })

  // A chunk can cut through an email, URL or path. Apply these exclusions to
  // the original text as well, so its fragments never become person names.
  const blocked = nonNameRanges(text)
  const found: Span[] = []
  for (let from = 0; from < tokens.length; from += CHUNK_STRIDE) {
    const to = Math.min(from + CHUNK_TOKENS, tokens.length)
    const start = from === 0 ? 0 : tokens[from].start
    const end = to === tokens.length ? text.length : tokens[to].start
    const spans = await veil.detect({
      ...resolved,
      context,
      text: text.slice(start, end),
      // These options apply to the whole answer, not to each chunk fragment.
      expectName: false,
      always: [],
      never: []
    })
    for (const span of spans) {
      if (
        !Number.isInteger(span.start) ||
        !Number.isInteger(span.end) ||
        span.start < 0 ||
        span.end > end - start ||
        span.start >= span.end
      )
        throw new Error("veil: invalid chunk offsets")
      const mapped = {
        ...span,
        start: start + span.start,
        end: start + span.end
      }
      if (
        mapped.type !== "PERSON" ||
        !blocked.some(([a, b]) => mapped.start < b && a < mapped.end)
      )
        found.push(mapped)
    }
    if (to === tokens.length) break
  }
  return applyTermOverrides(text, mergeSpans(found, false), resolved)
}
