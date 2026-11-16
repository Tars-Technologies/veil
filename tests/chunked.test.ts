import { readFileSync } from "node:fs"
import { describe, expect, it, vi } from "vitest"
import { detectChunked } from "../src/chunked"
import { MAX_TOKENS, type Span, tokenize } from "../src/engine"
import { loadModel } from "../src/model"
import { createVeil } from "../src/veil"

const MODEL_FILE = new URL("../veil-0.3.1.bin", import.meta.url)

function names(text: string): Span[] {
  return [...text.matchAll(/John Smith|Anna Schmidt/gu)].map((match) => ({
    start: match.index,
    end: match.index + match[0].length,
    type: "PERSON"
  }))
}

describe("bounded name detection", () => {
  it("keeps ordinary messages in one call and skips empty inputs", async () => {
    const detect = vi.fn(async ({ text }: { text: string }) => names(text))
    expect(await detectChunked({ detect }, "")).toEqual([])
    expect(detect).not.toHaveBeenCalled()
    const text = "🙂 Contact John Smith"
    expect(await detectChunked({ detect }, text)).toEqual(names(text))
    expect(detect).toHaveBeenCalledExactlyOnceWith({ text })
  })

  it.each([
    2303, 2304, 2330, 2367, 2368, 4607, 4671
  ])("preserves names around chunk boundaries at token %i", async (position) => {
    const text = `${". ".repeat(position)}John Smith ${". ".repeat(5000 - position)}Anna Schmidt`
    let active = 0
    let maxActive = 0
    const detect = vi.fn(async ({ text }: { text: string }) => {
      active++
      maxActive = Math.max(maxActive, active)
      expect(tokenize(text).length).toBeLessThanOrEqual(MAX_TOKENS)
      await Promise.resolve()
      active--
      return names(text)
    })
    expect(await detectChunked({ detect }, text)).toEqual(names(text))
    expect(maxActive).toBe(1)
    expect(detect.mock.calls.length).toBeGreaterThan(1)
  })

  it("counts punctuation and initials with the model tokenizer and preserves UTF-16", async () => {
    const text = `${"🙂 J. ".repeat(1600)}Call John Smith and Anna Schmidt`
    const detect = vi.fn(async ({ text }: { text: string }) => {
      expect(tokenize(text).length).toBeLessThanOrEqual(MAX_TOKENS)
      expect(text).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/u)
      return names(text)
    })
    expect(await detectChunked({ detect }, text)).toEqual(names(text))
  })

  it("merges partially overlapping names without duplicating them", async () => {
    const text = `${". ".repeat(2320)}John Smith ${". ".repeat(900)}`
    let call = 0
    const detect = vi.fn(async ({ text }: { text: string }) => {
      const found = names(text)
      if (++call === 1) return found.map((s) => ({ ...s, end: s.start + 4 }))
      return found
    })
    expect(await detectChunked({ detect }, text)).toEqual(names(text))
  })

  it("does not reinterpret fragments of URLs and paths as names", async () => {
    const text = `https://example.com/${"a/".repeat(1600)}John-Smith`
    const detect = vi.fn(
      async ({ text }: { text: string }): Promise<Span[]> => {
        const start = text.indexOf("John-Smith")
        return start < 0 ? [] : [{ start, end: start + 10, type: "PERSON" }]
      }
    )
    expect(await detectChunked({ detect }, text)).toEqual([])
  })

  it("rejects the entire result if a later chunk fails", async () => {
    const detect = vi
      .fn(async ({ text }: { text: string }) => names(text))
      .mockImplementationOnce(async ({ text }) => names(text))
      .mockRejectedValueOnce(new Error("model failed"))
    await expect(
      detectChunked({ detect }, `John Smith ${". ".repeat(3500)}`)
    ).rejects.toThrow("model failed")
  })

  it("rejects invalid offsets in a chunk rather than mapping into another chunk", async () => {
    const detect = vi.fn(
      async ({ text }: { text: string }): Promise<Span[]> => [
        { start: 0, end: text.length + 1, type: "PERSON" }
      ]
    )
    await expect(detectChunked({ detect }, ". ".repeat(3500))).rejects.toThrow(
      "invalid chunk offsets"
    )
  })

  it("retains the total character bound", async () => {
    const detect = vi.fn(async () => [])
    await expect(detectChunked({ detect }, "x".repeat(20001))).rejects.toThrow(
      "20000"
    )
    expect(detect).not.toHaveBeenCalled()
  })

  it("runs the shipped model on dense input that exceeds its per-call token limit", async () => {
    const veil = createVeil({ model: loadModel() })
    const text = `${". ".repeat(3500)}🙂 Contact John Smith`
    await expect(veil.detect({ text })).rejects.toThrow("3000")
    const spans = await detectChunked(veil, text)
    expect(spans.map(({ start, end }) => text.slice(start, end))).toContain(
      "John Smith"
    )
    expect(
      spans.find((s) => text.slice(s.start, s.end) === "John Smith")?.start
    ).toBe(text.indexOf("John Smith"))
  })

  it("includes question tokens in the per-call budget", async () => {
    const context = ". ".repeat(150)
    const text = `${". ".repeat(2950)}John Smith`
    const veil = createVeil({ model: loadModel() })
    await expect(veil.detect({ text, context })).rejects.toThrow("3000")
    const spans = await detectChunked(veil, text, { context })
    expect(spans.map(({ start, end }) => text.slice(start, end))).toContain(
      "John Smith"
    )
  })

  it("applies exact term overrides to the original input across chunk boundaries", async () => {
    const term = `Long ${"word ".repeat(70)}Name`
    const text = `${". ".repeat(2320)}${term} ${". ".repeat(900)}`
    const detect = vi.fn(
      async (_options: import("../src/detect").MaskOptions) => []
    )
    const spans = await detectChunked({ detect }, text, {
      always: [term],
      context: "Question",
      expectName: true
    })
    expect(spans.map(({ start, end }) => text.slice(start, end))).toEqual([
      term
    ])
    expect(
      detect.mock.calls.every(([options]) => options.expectName === false)
    ).toBe(true)
  })

  it("preserves defaults and overrides through bounded calls", async () => {
    const text = `${". ".repeat(3500)}John Smith`
    const veil = createVeil({
      model: loadModel(),
      defaults: { never: ["John Smith"], context: "Who should I contact?" }
    })
    expect(await detectChunked(veil, text)).toEqual([])
    const spans = await detectChunked(veil, text, {
      never: [],
      always: ["John Smith"]
    })
    expect(spans.map(({ start, end }) => text.slice(start, end))).toContain(
      "John Smith"
    )
  })

  it.each([
    "اسمي محمد أحمد وأريد حجز موعد",
    "मेरा नाम राहुल शर्मा है",
    "Меня зовут Иван Петров",
    "Με λένε Γιάννης Παπαδόπουλος"
  ])("does not claim model coverage for non-Latin scripts: %s", async (text) => {
    expect(
      await detectChunked(createVeil({ model: loadModel() }), text)
    ).toEqual([])
  })

  it("embeds exactly the shipped model bytes using the portable decoder", () => {
    const descriptor = Object.getOwnPropertyDescriptor(Uint8Array, "fromBase64")
    try {
      Object.defineProperty(Uint8Array, "fromBase64", {
        configurable: true,
        value: undefined
      })
      const expected = readFileSync(MODEL_FILE)
      expect(Buffer.from(loadModel()).equals(expected)).toBe(true)
    } finally {
      if (descriptor)
        Object.defineProperty(Uint8Array, "fromBase64", descriptor)
      else Reflect.deleteProperty(Uint8Array, "fromBase64")
    }
  })

  it("uses native decoding when available and produces the same model bytes", () => {
    const descriptor = Object.getOwnPropertyDescriptor(Uint8Array, "fromBase64")
    const native = vi.fn(
      (base64: string) => new Uint8Array(Buffer.from(base64, "base64"))
    )
    try {
      Object.defineProperty(Uint8Array, "fromBase64", {
        configurable: true,
        value: native
      })
      const expected = readFileSync(MODEL_FILE)
      expect(Buffer.from(loadModel()).equals(expected)).toBe(true)
      expect(native).toHaveBeenCalledTimes(1)
    } finally {
      if (descriptor)
        Object.defineProperty(Uint8Array, "fromBase64", descriptor)
      else Reflect.deleteProperty(Uint8Array, "fromBase64")
    }
  })
})
