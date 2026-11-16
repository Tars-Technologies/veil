// Runs the Veil model: text in, typed spans of personal data out. Pure TypeScript, no dependencies.
// The feature code must match the training code exactly; do not change it without retraining.

export const SPAN_TYPES = [
  "PERSON",
  "ADDRESS",
  "DOB",
  "ID",
  "FINANCIAL",
  "PHONE",
  "EMAIL"
] as const
export type SpanType = (typeof SPAN_TYPES)[number]
export type Span = { start: number; end: number; type: SpanType }
export type ModelInfo = { name: string; version: string; format: number }

type Tensor = { offset: number; shape: number[] }
type Header = ModelInfo & {
  config: {
    buckets: number
    hidden: number
    layers: number
    types: string[]
    bias: Record<string, number>
    lists: number
    bloom: number
    levels: number[]
    emb_length_range: [low: number, high: number]
  }
  tensors: Record<string, Tensor>
}
type Token = { start: number; end: number; word: string }
export type Range = [start: number, end: number]

const FORMAT = 3
export const CONTEXT_CHARS = 300
// Longer inputs are refused, so the cost of one call has a limit; the caller decides what to do with such a text.
export const MAX_CHARS = 20000
export const MAX_TOKENS = 3000
export const WINDOW_TOKENS = 256
export const WINDOW_STRIDE = 192
const SHAPE_WIDTH = 16
const SHAPES = 21
const COMMON_WORD = 4

const TOKEN =
  /[\p{L}\p{M}\p{N}]+(?:[-‐‑][\p{L}\p{M}\p{N}]+)*|[^\s\p{L}\p{M}\p{N}]/gu
const WORDLIKE = /[\p{L}\p{N}]/u
const LATIN = /\p{Script=Latin}/u
const LETTER = /\p{L}/u
const DIGIT = /\p{Nd}/u
const DIGITS = /\p{Nd}/gu
const ASCII_DIGITS = /^[0-9]+$/
const RUNS = /(.)\1+/gsu
const BOUNDARY_PUNCT = new Set([".", "!", "?", ":", "-", "—", ","])

export function tokenize(text: string): Token[] {
  const out: Token[] = []
  for (const match of text.matchAll(TOKEN)) {
    const start = match.index
    let end = start + match[0].length
    // An initial keeps its full stop: "J." is one token.
    if (
      WORDLIKE.test(match[0]) &&
      [...match[0]].length === 1 &&
      text[end] === "."
    )
      end += 1
    else if (
      !WORDLIKE.test(match[0]) &&
      out.length &&
      out[out.length - 1].end > start
    )
      continue
    out.push({ start, end, word: text.slice(start, end) })
  }
  return out
}

const isUpper = (c: string) => c.toLowerCase() !== c

function shapeId(
  word: string,
  index: number,
  previous: string | null,
  afterNewline: boolean
): number {
  const letters = [...word].filter((c) => LETTER.test(c))
  let kind: number
  if (!letters.length) kind = DIGIT.test(word) ? 0 : 1
  else {
    const upper = letters.filter(isUpper).length
    if (upper === 0) kind = 2
    else if (upper === letters.length) kind = letters.length > 1 ? 3 : 4
    else if (upper === 1 && isUpper(letters[0])) kind = 5
    else kind = 6
  }
  const position =
    index === 0 || afterNewline
      ? 0
      : previous !== null && BOUNDARY_PUNCT.has(previous)
        ? 1
        : 2
  return kind * 3 + position
}

function wordShape(word: string): string {
  let out = ""
  for (const c of word)
    out += DIGIT.test(c) ? "0" : LETTER.test(c) ? (isUpper(c) ? "A" : "a") : c
  return out
}

function hashChars(text: string, from = 0, to = text.length): number {
  let hash = 0x811c9dc5
  for (let i = from; i < to; i++)
    hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0
  return hash
}

const utf8 = new TextEncoder()

function hashBytes(text: string): number {
  let hash = 0x811c9dc5
  for (const byte of utf8.encode(text))
    hash = Math.imul(hash ^ byte, 0x01000193) >>> 0
  return hash
}

type PackedTable = {
  codes: Uint8Array
  lengthCodes: Uint8Array
  lengths: Float32Array
  rotation: Float32Array
  levels: number[]
  scaled: Float64Array
  sum: Float32Array
}

function addRow(table: PackedTable, row: number, weight: number): void {
  const { codes, levels, scaled, sum } = table
  const scale = table.lengths[table.lengthCodes[row]] * weight
  for (let level = 0; level < 4; level++) scaled[level] = levels[level] * scale
  const start = (row * sum.length) >> 2
  for (let k = 0; k < sum.length; k += 4) {
    const byte = codes[start + (k >> 2)]
    sum[k] += scaled[byte & 3]
    sum[k + 1] += scaled[(byte >> 2) & 3]
    sum[k + 2] += scaled[(byte >> 4) & 3]
    sum[k + 3] += scaled[byte >> 6]
  }
}

function addSum(table: PackedTable, out: Float32Array, at: number): void {
  const { rotation, sum } = table
  const width = sum.length
  const unit = 1 / Math.sqrt(width)
  for (let k = 0; k < width; k++) {
    const value = sum[k] * unit
    if (value !== 0)
      for (let j = 0; j < width; j++)
        out[at + j] += value * rotation[k * width + j]
    sum[k] = 0
  }
}

type Kernel = {
  memory: { buffer: ArrayBuffer }
  /** out[g] = b[g] + W_ih[g]·x + W_hh[g]·h for each of `rows` gates; all arguments are byte offsets into `memory`. */
  gates: (
    wih: number,
    x: number,
    inWidth: number,
    whh: number,
    h: number,
    hWidth: number,
    b: number,
    out: number,
    rows: number
  ) => void
}

const GATES_WASM =
  "AGFzbQEAAAABDQFgCX9/f39/f39/fwADAgEABQMBABAHEgIGbWVtb3J5AgAFZ2F0ZXMAAArwAQHtAQQCfwF7AX8BfQJAA0AgCSAITw0BQwAAAAD9EyELIAAgCSACbEECdGohDEEAIQoCQANAIAogAk8NASALIAwgCkECdGr9AAQAIAEgCkECdGr9AAQA/eYB/eQBIQsgCkEEaiEKDAALCyADIAkgBWxBAnRqIQxBACEKAkADQCAKIAVPDQEgCyAMIApBAnRq/QAEACAEIApBAnRq/QAEAP3mAf3kASELIApBBGohCgwACwsgC/0fACAL/R8BkiAL/R8CIAv9HwOSkiENIAcgCUECdGogDSAGIAlBAnRqKgIAkjgCACAJQQFqIQkMAAsLCw=="

/** The wasm SIMD kernel, or null where WebAssembly or its SIMD extension is not available. */
function simdKernel(pages: number): Kernel | null {
  if (typeof WebAssembly === "undefined") return null
  try {
    const bytes = Uint8Array.from(atob(GATES_WASM), (c) => c.charCodeAt(0))
    const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {})
    const memory = instance.exports.memory as WebAssembly.Memory
    const current = memory.buffer.byteLength / 65536
    if (pages > current) memory.grow(pages - current)
    return { memory, gates: instance.exports.gates as Kernel["gates"] }
  } catch {
    return null
  }
}

/** Plain-JS copy of the SIMD kernel. It keeps four float32 lanes and the same order of additions, so results are bit-identical. */
function scalarKernel(pages: number): Kernel {
  const buffer = new ArrayBuffer(pages * 65536)
  const f = new Float32Array(buffer)
  const r = Math.fround
  return {
    memory: { buffer },
    gates: (wih, x, inWidth, whh, h, hWidth, b, out, rows) => {
      const xi = x >> 2
      const hi = h >> 2
      for (let g = 0; g < rows; g++) {
        let a0 = 0,
          a1 = 0,
          a2 = 0,
          a3 = 0
        let row = (wih >> 2) + g * inWidth
        for (let k = 0; k < inWidth; k += 4) {
          a0 = r(a0 + r(f[row + k] * f[xi + k]))
          a1 = r(a1 + r(f[row + k + 1] * f[xi + k + 1]))
          a2 = r(a2 + r(f[row + k + 2] * f[xi + k + 2]))
          a3 = r(a3 + r(f[row + k + 3] * f[xi + k + 3]))
        }
        row = (whh >> 2) + g * hWidth
        for (let k = 0; k < hWidth; k += 4) {
          a0 = r(a0 + r(f[row + k] * f[hi + k]))
          a1 = r(a1 + r(f[row + k + 1] * f[hi + k + 1]))
          a2 = r(a2 + r(f[row + k + 2] * f[hi + k + 2]))
          a3 = r(a3 + r(f[row + k + 3] * f[hi + k + 3]))
        }
        f[(out >> 2) + g] = r(r(r(a0 + a1) + r(a2 + a3)) + f[(b >> 2) + g])
      }
    }
  }
}

/**
 * Joins spans that overlap, and with `joinTouching` also spans of one type that touch. The type of the first span is kept.
 */
export function mergeSpans(spans: Span[], joinTouching: boolean): Span[] {
  const merged: Span[] = []
  for (const span of [...spans].sort((a, b) => a.start - b.start)) {
    const last = merged[merged.length - 1]
    if (
      last &&
      (span.start < last.end ||
        (joinTouching && span.start === last.end && span.type === last.type))
    )
      last.end = Math.max(last.end, span.end)
    else merged.push({ ...span })
  }
  return merged
}

type Lstm = { wih: number; whh: number; b: number; inWidth: number }

const CURVE_STEPS = 1 << 14
const CURVE_RANGE = 10
const curve = (value: (x: number) => number) =>
  Float32Array.from({ length: CURVE_STEPS + 1 }, (_, index) =>
    value(((index / CURVE_STEPS) * 2 - 1) * CURVE_RANGE)
  )
const curveIndex = (x: number) =>
  x <= -CURVE_RANGE
    ? 0
    : x >= CURVE_RANGE
      ? CURVE_STEPS
      : Math.floor((x + CURVE_RANGE) * (CURVE_STEPS / (2 * CURVE_RANGE)) + 0.5)

function readHeader(bytes: ArrayBuffer): { header: Header; base: number } {
  let header: Header | undefined
  let base = 0
  try {
    const length = new DataView(bytes).getUint32(0, true)
    header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(bytes, 4, length))
    )
    base = 4 + length
  } catch {}
  if (header?.name !== "veil") throw new Error("veil: not a Veil model file")
  if (header.format !== FORMAT)
    throw new Error(
      `veil: this library reads model format ${FORMAT}; the file has format ${header.format}`
    )
  const body = new Uint8Array(bytes, 0, bytes.byteLength - 4)
  let hash = 0x811c9dc5
  for (let i = 0; i < body.length; i++)
    hash = Math.imul(hash ^ body[i], 0x01000193) >>> 0
  if (hash !== new DataView(bytes).getUint32(body.length, true))
    throw new Error("veil: the model file is damaged")
  return { header, base }
}

export class Engine {
  readonly model: ModelInfo
  private readonly buckets: number
  private readonly hidden: number
  private readonly width: number
  private readonly types: string[]
  private readonly bias: Record<string, number>
  private readonly labels: number
  private readonly free: boolean[]
  private readonly allowed: boolean[][]
  private readonly chars: PackedTable
  private readonly shapes: Float32Array
  private readonly lists: Uint8Array[]
  private readonly probes: number
  private readonly lstm: Lstm[][]
  private readonly kernel: Kernel
  private readonly buffers: { x: number; y: number[]; h: number; gates: number }
  private readonly outW: Float32Array
  private readonly outB: Float32Array
  private readonly sigmoid = curve((x) => 1 / (1 + Math.exp(-x)))
  private readonly tanh = curve(Math.tanh)

  constructor(bytes: ArrayBuffer) {
    const { header, base } = readHeader(bytes)
    this.model = {
      name: header.name,
      version: header.version,
      format: header.format
    }
    const tensor = (name: string) => {
      const found = header.tensors[name]
      if (!found) throw new Error(`veil: the model file has no "${name}"`)
      return {
        at: base + found.offset,
        shape: found.shape,
        size: found.shape.reduce((a, b) => a * b, 1)
      }
    }
    const floats = (name: string) => {
      if (header.tensors[name])
        return new Float32Array(bytes, tensor(name).at, tensor(name).size)
      // Stored as int8 with one scale per row; expanded once at load.
      const {
        at,
        shape: [rows, columns]
      } = tensor(`${name}_q`)
      const values = new Int8Array(bytes, at, rows * columns)
      const scales = new Float32Array(bytes, tensor(`${name}_scale`).at, rows)
      const out = new Float32Array(rows * columns)
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < columns; c++)
          out[r * columns + c] = values[r * columns + c] * scales[r]
      return out
    }

    const { buckets, hidden, layers, levels, types } = header.config
    const [low, high] = header.config.emb_length_range
    const codes = tensor("emb_codes")
    const [rows, bytesPerRow] = codes.shape
    this.chars = {
      codes: new Uint8Array(bytes, codes.at, codes.size),
      lengthCodes: new Uint8Array(bytes, tensor("emb_length").at, rows),
      lengths: Float32Array.from({ length: 256 }, (_, code) =>
        Math.exp(low + (code / 255) * (high - low))
      ),
      rotation: floats("emb_rotation"),
      levels,
      scaled: new Float64Array(4),
      sum: new Float32Array(4 * bytesPerRow)
    }
    this.buckets = buckets
    this.hidden = hidden
    this.types = types
    this.bias = header.config.bias
    // Labels: 0 outside; then begin, inside, last, single for each type.
    this.labels = 1 + 4 * types.length
    this.free = Array.from(
      { length: this.labels },
      (_, label) =>
        label === 0 || (label - 1) % 4 === 0 || (label - 1) % 4 === 3
    )
    this.allowed = Array.from({ length: this.labels }, (_, previous) => {
      const position = (previous - 1) % 4
      return Array.from({ length: this.labels }, (_, label) =>
        previous !== 0 && position <= 1
          ? label === previous - position + 1 ||
            label === previous - position + 2
          : this.free[label]
      )
    })
    this.shapes = floats("shape")
    // Known person names, places and common words, each as a Bloom filter over the lower-case word.
    this.lists = Array.from({ length: header.config.lists }, (_, bit) => {
      const { at, size } = tensor(`bloom_${bit}`)
      return new Uint8Array(bytes, at, size)
    })
    this.probes = header.config.bloom
    this.width = this.chars.sum.length + SHAPE_WIDTH

    const names: string[] = []
    for (let layer = 0; layer < layers; layer++)
      for (const direction of ["f", "b"])
        for (const part of ["wih", "whh", "b"])
          names.push(`l${layer}${direction}_${part}`)
    const weights = new Map(names.map((name) => [name, floats(name)]))
    const weightFloats = [...weights.values()].reduce(
      (a, values) => a + values.length,
      0
    )
    const bufferFloats =
      WINDOW_TOKENS * this.width + 2 * WINDOW_TOKENS * 2 * hidden + 5 * hidden
    const pages = Math.ceil(((weightFloats + bufferFloats) * 4) / 65536) + 1
    this.kernel = simdKernel(pages) ?? scalarKernel(pages)
    let cursor = 0
    const reserve = (count: number) => {
      const at = cursor
      cursor += count * 4
      return at
    }
    const place = (name: string) => {
      const values = weights.get(name)!
      const at = reserve(values.length)
      new Float32Array(this.kernel.memory.buffer, at, values.length).set(values)
      return at
    }
    this.lstm = Array.from({ length: layers }, (_, layer) =>
      ["f", "b"].map((direction) => ({
        wih: place(`l${layer}${direction}_wih`),
        whh: place(`l${layer}${direction}_whh`),
        b: place(`l${layer}${direction}_b`),
        inWidth: layer === 0 ? this.width : 2 * hidden
      }))
    )
    this.buffers = {
      x: reserve(WINDOW_TOKENS * this.width),
      y: [
        reserve(WINDOW_TOKENS * 2 * hidden),
        reserve(WINDOW_TOKENS * 2 * hidden)
      ],
      h: reserve(hidden),
      gates: reserve(4 * hidden)
    }
    this.outW = floats("out_w")
    this.outB = floats("out_b")
  }

  /**
   * Spans of the given `types` in `text`, read with `context` (the text that came before) in front of it.
   * `bias` is added to the person labels before the final choice: above 0 masks more names, below 0 masks fewer.
   * Throws on a text that is too long and on non-finite numbers, so a caller never gets a silent "nothing found".
   */
  detect(
    text: string,
    context = "",
    bias = 0,
    types: readonly SpanType[] = SPAN_TYPES
  ): Span[] {
    if (!Number.isFinite(bias))
      throw new Error("veil: bias is not a finite number")
    let tail = context.slice(-CONTEXT_CHARS)
    // Do not start the context in the middle of a surrogate pair.
    if (/^[\uDC00-\uDFFF]/.test(tail)) tail = tail.slice(1)
    const prefix = tail ? `${tail}\n` : ""
    const full = prefix + text
    if (full.length > MAX_CHARS)
      throw new Error(`veil: text longer than ${MAX_CHARS} characters`)
    const tokens = tokenize(full)
    if (tokens.length > MAX_TOKENS)
      throw new Error(`veil: text longer than ${MAX_TOKENS} tokens`)

    // Without a Latin letter no name can pass the script rule, so the model is not run for names alone.
    if (types.every((type) => type === "PERSON") && !LATIN.test(text)) return []

    const biases = new Float64Array(this.labels)
    for (const [index, type] of this.types.entries())
      biases.fill(
        (this.bias[type] ?? 0) + (type === "PERSON" ? bias : 0),
        1 + 4 * index,
        5 + 4 * index
      )
    // Emails, links and paths are found once per call, and only when a window holds a name.
    let blocked: Range[] | undefined
    const nonNames = () => {
      blocked ??= nonNameRanges(full)
      return blocked
    }
    const found: Span[] = []
    for (let from = 0; from < tokens.length; from += WINDOW_STRIDE) {
      found.push(
        ...this.detectWindow(
          full,
          tokens.slice(from, from + WINDOW_TOKENS),
          biases,
          nonNames
        )
      )
      if (from + WINDOW_TOKENS >= tokens.length) break
    }
    return mergeSpans(
      found.filter((span) => types.includes(span.type)),
      true
    )
      .filter((span) => span.end > prefix.length)
      .map((span) => ({
        start: Math.max(0, span.start - prefix.length),
        end: span.end - prefix.length,
        type: span.type
      }))
  }

  /** True when the word is on the model's list of common words. */
  isCommonWord(word: string): boolean {
    return (this.flags(hashBytes(word.toLowerCase())) & COMMON_WORD) !== 0
  }

  private flags(hash: number): number {
    const step = Math.imul(hash, 0x9e3779b1) | 1
    let flags = 0
    for (let bit = 0; bit < this.lists.length; bit++) {
      const list = this.lists[bit]
      const bits = list.length * 8
      let hit = true
      for (let probe = 0; probe < this.probes && hit; probe++) {
        const at = ((hash + Math.imul(probe, step)) >>> 0) % bits
        hit = ((list[at >> 3] >> (at & 7)) & 1) === 1
      }
      if (hit) flags |= 1 << bit
    }
    return flags
  }

  /** Adds the mean of the hashed feature vectors of a word to `out`: character n-grams, and for a word with digits its shape and length. */
  private embedChars(word: string, out: Float32Array, at: number): void {
    const hasDigit = DIGIT.test(word)
    const lowered = hasDigit
      ? word.toLowerCase().replace(DIGITS, "0")
      : word.toLowerCase()
    const marked = `<${lowered}>`
    const ids = [hashChars(`w:${lowered}`)]
    for (const n of [2, 3, 4, 5])
      for (let i = 0; i + n <= marked.length; i++)
        ids.push(hashChars(marked, i, i + n))
    if (hasDigit) {
      const shape = wordShape(word)
      const letters = [...word].filter((c) => LETTER.test(c))
      const kind = !letters.length ? "d" : letters.every(isUpper) ? "u" : "x"
      ids.push(
        hashChars(`s:${[...shape].slice(0, 16).join("")}`),
        hashChars(`c:${[...shape.replace(RUNS, "$1")].slice(0, 16).join("")}`),
        hashChars(`k:${kind}${Math.min([...word].length, 20)}`)
      )
      if (ASCII_DIGITS.test(word)) {
        ids.push(hashChars(`l:${Math.min(word.length, 20)}`))
        if (word.length <= 2) ids.push(hashChars(`v:${Number(word)}`))
        else if (word.length === 4) ids.push(hashChars(`y:${word.slice(0, 3)}`))
      }
    }
    const weight = 1 / ids.length
    for (const id of ids) addRow(this.chars, id % this.buckets, weight)
    addSum(this.chars, out, at)
  }

  private runLstm(
    input: number,
    steps: number,
    weights: Lstm,
    reverse: boolean,
    output: number
  ): void {
    const H = this.hidden
    const memory = this.kernel.memory.buffer
    const h = new Float32Array(memory, this.buffers.h, H)
    const gates = new Float32Array(memory, this.buffers.gates, 4 * H)
    const out = new Float32Array(memory, output, steps * 2 * H)
    const c = new Float32Array(H)
    const sigmoid = this.sigmoid
    const tanh = this.tanh
    h.fill(0)
    const offset = reverse ? H : 0
    for (let s = 0; s < steps; s++) {
      const t = reverse ? steps - 1 - s : s
      this.kernel.gates(
        weights.wih,
        input + t * weights.inWidth * 4,
        weights.inWidth,
        weights.whh,
        this.buffers.h,
        H,
        weights.b,
        this.buffers.gates,
        4 * H
      )
      const base = t * 2 * H + offset
      for (let k = 0; k < H; k++) {
        const i = sigmoid[curveIndex(gates[k])]
        const f = sigmoid[curveIndex(gates[H + k])]
        const g = tanh[curveIndex(gates[2 * H + k])]
        const o = sigmoid[curveIndex(gates[3 * H + k])]
        const cell = f * c[k] + i * g
        c[k] = cell
        const value = o * tanh[curveIndex(cell)]
        h[k] = value
        out[base + k] = value
      }
    }
  }

  private detectWindow(
    text: string,
    tokens: Token[],
    biases: Float64Array,
    nonNames: () => Range[]
  ): Span[] {
    const n = tokens.length
    if (!n) return []
    const { width, hidden: H, labels } = this
    const dim = this.chars.sum.length
    const x = new Float32Array(
      this.kernel.memory.buffer,
      this.buffers.x,
      n * width
    )
    x.fill(0)
    tokens.forEach((token, i) => {
      this.embedChars(token.word, x, i * width)
      const afterNewline =
        i > 0 && text.slice(tokens[i - 1].end, token.start).includes("\n")
      const shape =
        (shapeId(token.word, i, i ? tokens[i - 1].word : null, afterNewline) +
          SHAPES * this.flags(hashBytes(token.word.toLowerCase()))) *
        SHAPE_WIDTH
      x.set(this.shapes.subarray(shape, shape + SHAPE_WIDTH), i * width + dim)
    })

    let input = this.buffers.x
    this.lstm.forEach(([forward, backward], layer) => {
      const output = this.buffers.y[layer % 2]
      this.runLstm(input, n, forward, false, output)
      this.runLstm(input, n, backward, true, output)
      input = output
    })
    const state = new Float32Array(this.kernel.memory.buffer, input, n * 2 * H)

    // The best label sequence that obeys the begin/inside/last order of a span.
    let best = new Float64Array(labels)
    const back: Int32Array[] = []
    const row = new Float64Array(labels)
    for (let t = 0; t < n; t++) {
      let max = -Infinity
      for (let label = 0; label < labels; label++) {
        let sum = this.outB[label]
        for (let k = 0; k < 2 * H; k++)
          sum += this.outW[label * 2 * H + k] * state[t * 2 * H + k]
        row[label] = sum
        if (sum > max) max = sum
      }
      let total = 0
      for (let label = 0; label < labels; label++)
        total += Math.exp(row[label] - max)
      const log = Math.log(total) + max
      if (!Number.isFinite(log))
        throw new Error("veil: non-finite model output")
      for (let label = 0; label < labels; label++)
        row[label] = row[label] - log + biases[label]
      if (t === 0) {
        for (let label = 0; label < labels; label++)
          best[label] = this.free[label] ? row[label] : -1e9
        continue
      }
      const next = new Float64Array(labels)
      const pointers = new Int32Array(labels)
      for (let label = 0; label < labels; label++) {
        let value = -Infinity
        let pointer = 0
        for (let previous = 0; previous < labels; previous++) {
          const candidate =
            best[previous] + (this.allowed[previous][label] ? 0 : -1e9)
          if (candidate > value) {
            value = candidate
            pointer = previous
          }
        }
        next[label] = value + row[label]
        pointers[label] = pointer
      }
      best = next
      back.push(pointers)
    }
    let last = 0
    let lastValue = -Infinity
    for (let label = 0; label < labels; label++) {
      const canEnd = label === 0 || (label - 1) % 4 >= 2
      const value = best[label] + (canEnd ? 0 : -1e9)
      if (value > lastValue) [last, lastValue] = [label, value]
    }
    const path = [last]
    for (let i = back.length - 1; i >= 0; i--)
      path.push(back[i][path[path.length - 1]])
    path.reverse()

    const names: Range[] = []
    const out: Span[] = []
    let start: number | null = null
    tokens.forEach((token, i) => {
      const label = path[i]
      const position = (label - 1) % 4
      if (label === 0) start = null
      else if (position === 0) start = token.start
      if (label === 0 || position <= 1 || (position === 2 && start === null))
        return
      const from = position === 3 ? token.start : start!
      start = null
      const type = this.types[(label - 1) >> 2]
      if (type === "PERSON") names.push([from, token.end])
      else if (SPAN_TYPES.includes(type as SpanType))
        out.push({
          start: type === "PHONE" && text[from - 1] === "+" ? from - 1 : from,
          end: token.end,
          type: type as SpanType
        })
    })
    for (const [from, to] of dropNonNames(
      text,
      splitAtNewlines(text, names),
      nonNames
    ))
      out.push({ start: from, end: to, type: "PERSON" })
    return out
  }
}

function splitAtNewlines(text: string, ranges: Range[]): Range[] {
  const out: Range[] = []
  for (const [start, end] of ranges) {
    let cursor = start
    for (const line of text.slice(start, end).split("\n")) {
      const lead = line.length - line.trimStart().length
      const trimmed = line.trim()
      if (trimmed) out.push([cursor + lead, cursor + lead + trimmed.length])
      cursor += line.length + 1
    }
  }
  return out
}

// Policy: email addresses, URLs and file paths are not person names; never mask inside them.
// A path is: /two/or/more parts, a drive (C:\), a relative start (./ or ../) or a network share (\\host).
// One part after a slash ("/Anna", "and/or") is ordinary chat text and is not protected.
const URL_OR_PATH =
  /(?:https?:\/\/|www\.)[^\s<>"'()[\],;]*[^\s<>"'()[\],;.:!?]|(?<![\p{L}\p{N}.])\/[\p{L}\p{N}_.-]+(?:\/[\p{L}\p{N}_.-]+)+|(?<![\p{L}\p{N}.])(?:[a-z]:\\|\.{1,2}[\\/]|\\\\)[^\s<>"'()[\],;]+/giu
const EMAIL_PART = /[^\s<>"'()[\]@,;:]/u
const EMAIL_TAIL = /\.[A-Za-z]{2,}/g

// Email addresses, found from each "@" outwards. One pass over the text, where a single pattern would retry at every character.
function emailRanges(text: string): Range[] {
  const out: Range[] = []
  for (let at = text.indexOf("@"); at >= 0; at = text.indexOf("@", at + 1)) {
    let left = at
    while (left > 0 && EMAIL_PART.test(text[left - 1])) left--
    let right = at + 1
    while (right < text.length && EMAIL_PART.test(text[right])) right++
    if (left === at) continue
    let end = -1
    for (const tail of text.slice(at + 1, right).matchAll(EMAIL_TAIL))
      if (tail.index > 0) end = tail.index + tail[0].length
    if (end > 0 && !(out.length && left < out[out.length - 1][1]))
      out.push([left, at + 1 + end])
  }
  return out
}

/** Email addresses, links and file paths in `text`, in order. No name is masked inside them. */
export function nonNameRanges(text: string): Range[] {
  const ranges = [...text.matchAll(URL_OR_PATH)].map(
    (match): Range => [match.index, match.index + match[0].length]
  )
  for (const email of emailRanges(text))
    if (!ranges.some(([start, end]) => start <= email[0] && email[1] <= end))
      ranges.push(email)
  return ranges.sort((a, b) => a[0] - b[0])
}

// A word that ends in a digit or holds two or more digits is a handle or an identifier, not a name.
// A single inner digit (a typing error such as J0hn) is kept.
const isHandle = (word: string) =>
  (word.match(/\p{Nd}/gu) ?? []).length >= 2 || /\p{Nd}$/u.test(word)

function dropNonNames(
  text: string,
  ranges: Range[],
  nonNames: () => Range[]
): Range[] {
  if (!ranges.length) return []
  const blocked = nonNames()
  const clipped: Range[] = []
  for (const range of ranges) {
    // Cut the blocked parts out of a range; do not discard the whole range.
    let pieces: Range[] = [range]
    for (const [s, e] of blocked) {
      pieces = pieces.flatMap(([a, b]): Range[] =>
        b <= s || e <= a
          ? [[a, b]]
          : [
              ...(a < s ? [[a, s] as Range] : []),
              ...(e < b ? [[e, b] as Range] : [])
            ]
      )
    }
    for (const [a, b] of pieces) {
      const piece = text.slice(a, b)
      if (!WORDLIKE.test(piece)) continue
      // A cut can leave separators (" Doe", ",Jane"); keep the range on the word.
      const lead = piece.length - piece.replace(/^[\s,;:.]+/u, "").length
      const trail = piece.length - piece.replace(/[\s,;:]+$/u, "").length
      clipped.push([a + lead, b - trail])
    }
  }
  return clipped.filter(([start, end]) => {
    const piece = text.slice(start, end)
    // The model knows Latin-script names only; in other scripts it marks ordinary words.
    if (!LATIN.test(piece)) return false
    const words = piece.split(/\s+/).filter(Boolean)
    return !(words.length === 1 && isHandle(words[0]))
  })
}
