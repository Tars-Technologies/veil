import { readFileSync } from "node:fs"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createVeil as createBrowserVeil } from "../src/browser"
import { nonNameRanges, SPAN_TYPES } from "../src/engine"
import { createVeil, type MaskOptions } from "../src/veil"

type Recorded = MaskOptions & { spans: [number, number][] }

const read = (path: string) => readFileSync(new URL(path, import.meta.url))
const file = read("../veil-0.3.1.bin")
const model = file.buffer.slice(
  file.byteOffset,
  file.byteOffset + file.byteLength
) as ArrayBuffer
const recorded: Recorded[] = read("fixtures/recorded.jsonl")
  .toString()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line))

async function mismatches(rows = recorded) {
  const veil = createVeil({ model })
  let count = 0
  for (const { spans, ...options } of rows) {
    const found = (await veil.detect(options)).map((s) => [s.start, s.end])
    if (JSON.stringify(found) !== JSON.stringify(spans)) count++
  }
  return count
}

afterEach(() => vi.unstubAllGlobals())

describe("model output", () => {
  it("matches the recorded spans", async () => {
    expect(recorded).toHaveLength(300)
    expect(await mismatches()).toBe(0)
  })

  it("matches the recorded spans without WebAssembly", async () => {
    // The plain-JS path is about 6 times slower, so it runs a part of the rows.
    vi.stubGlobal("WebAssembly", undefined)
    expect(await mismatches(recorded.slice(0, 60))).toBe(0)
  })
})

describe("createVeil", () => {
  it("loads on first use and reports the model version", async () => {
    const veil = createVeil({ model })
    expect(veil.status).toBe("idle")
    expect(veil.model).toBeNull()
    expect(
      await veil.mask({ text: "my manager Daniel Okafor told me to call" })
    ).toBe("my manager [NAME] told me to call")
    expect(veil.status).toBe("ready")
    expect(veil.model).toEqual({ name: "veil", version: "0.3.1", format: 3 })
  })

  it("rejects a file that is not a Veil model", async () => {
    const veil = createVeil({ model: new ArrayBuffer(64) })
    await expect(veil.ready()).rejects.toThrow("not a Veil model file")
    expect(veil.status).toBe("error")
  })

  it("rejects a model file with one changed byte", async () => {
    const damaged = new Uint8Array(model.slice(0))
    damaged[damaged.length >> 1] ^= 1
    const veil = createVeil({ model: damaged.buffer })
    await expect(veil.ready()).rejects.toThrow("damaged")
  })

  it("applies defaults and lets a call override them", async () => {
    const veil = createVeil({
      model,
      defaults: { never: ["Ava"], replacement: "***" }
    })
    const text = "thanks Ava, please call Anna Schmidt"
    expect(await veil.mask({ text })).toBe("thanks Ava, please call ***")
    expect(await veil.mask({ text, never: [], replacement: "[X]" })).toBe(
      "thanks [X], please call [X]"
    )
  })
})

describe("defaults", () => {
  it("cannot change after the client is created", async () => {
    const given = { always: ["zeynep"] }
    const veil = createVeil({ model, defaults: given })
    given.always.length = 0
    expect(() => {
      ;(veil.defaults as { always?: string[] }).always = []
    }).toThrow()
    expect(() => veil.defaults.always?.push("anna")).toThrow()
    expect(await veil.mask({ text: "thank you zeynep" })).toBe(
      "thank you [NAME]"
    )
  })
})

describe("browser client", () => {
  it("resolves a relative model URL against the page, not the worker", async () => {
    const requests: { model?: unknown }[] = []
    class FakeWorker {
      onmessage: ((event: { data: unknown }) => void) | null = null
      onerror: unknown = null
      postMessage(request: { id: number; model?: unknown }) {
        requests.push(request)
        this.onmessage?.({
          data: {
            id: request.id,
            result: { name: "veil", version: "0", format: 1 }
          }
        })
      }
      terminate() {}
    }
    vi.stubGlobal("Worker", FakeWorker)
    vi.stubGlobal("document", { baseURI: "https://app.example/chat/page" })

    await createBrowserVeil({ model: "./veil.bin" }).ready()
    await createBrowserVeil({ model: "/veil.bin" }).ready()

    expect(requests.map((request) => request.model)).toEqual([
      "https://app.example/chat/veil.bin",
      "https://app.example/veil.bin"
    ])
  })
})

describe("protected ranges", () => {
  it("match the shared cases", () => {
    const cases: { text: string; ranges: [number, number][] }[] = JSON.parse(
      read("fixtures/protected_ranges.json").toString()
    )
    for (const { text, ranges } of cases) {
      expect(nonNameRanges(text), text).toEqual(ranges)
    }
  })
})

describe("options", () => {
  it("uses the options as they were at the call", async () => {
    const options = { text: "please call Anna Schmidt" }
    const masked = createVeil({ model }).mask(options)
    options.text = "hello"
    expect(await masked).toBe("please call [NAME]")
  })

  it("uses the lists as they were at the call", async () => {
    const always = ["zeynep"]
    const masked = createVeil({ model }).mask({
      text: "thank you zeynep",
      always
    })
    always.length = 0
    expect(await masked).toBe("thank you [NAME]")
  })

  it("keeps a default when the call gives undefined", async () => {
    const client = createVeil({ model, defaults: { replacement: "***" } })
    expect(
      await client.mask({ text: "call Anna Schmidt", replacement: undefined })
    ).toBe("call ***")
  })

  const veil = createVeil({ model })

  it("never: leaves a span only when it is exactly a listed term", async () => {
    const text = "please call Anna Schmidt"
    expect(await veil.mask({ text, never: ["anna schmidt"] })).toBe(text)
    expect(await veil.mask({ text, never: ["Anna"] })).toBe(
      "please call [NAME]"
    )
  })

  it("always: masks a listed term the model misses, as a whole word", async () => {
    const text = "thank you kestrel, anything else? kestrels"
    expect(await veil.mask({ text })).toBe(text)
    expect(await veil.mask({ text, always: ["Kestrel"] })).toBe(
      "thank you [NAME], anything else? kestrels"
    )
  })

  it("always: joins with an overlapping model span", async () => {
    expect(
      await veil.detect({ text: "call Anna Schmidt", always: ["Anna"] })
    ).toEqual([{ start: 5, end: 17, type: "PERSON" }])
  })

  it("expectName: masks a short answer as a whole", async () => {
    const context = "What is your name?"
    expect(await veil.mask({ text: "River", context })).toBe("River")
    expect(
      await veil.mask({ text: " River ", context, expectName: true })
    ).toBe(" [NAME] ")
  })

  it("always: does not mask inside an email address or a link", async () => {
    const text = "Anna wrote from anna@example.com and https://example.com/anna"
    expect(await veil.mask({ text, always: ["Anna"], threshold: 0.99 })).toBe(
      "[NAME] wrote from anna@example.com and https://example.com/anna"
    )
  })

  it("always: leaves upper-case links and every path form", async () => {
    const text =
      "see HTTPS://Anna.example.com, C:\\Anna\\file.txt, ./Anna.txt, ../docs/Anna, _/Anna/file_ and \\\\host\\Anna"
    expect(await veil.mask({ text, always: ["Anna"] })).toBe(text)
  })

  it("always: an ellipsis before a slash is not a path", async () => {
    expect(await veil.mask({ text: "call.../Anna", always: ["Anna"] })).toBe(
      "call.../[NAME]"
    )
  })

  it("always: a combining mark is part of the word", async () => {
    const text = "qz\u0301xw and qz"
    expect(await veil.mask({ text, always: ["qz"] })).toBe(
      "qz\u0301xw and [NAME]"
    )
  })

  it("expectName: a term on never is left", async () => {
    expect(
      await veil.mask({ text: "Zeynep", expectName: true, never: ["zeynep"] })
    ).toBe("Zeynep")
  })

  it("expectName: a term on both lists is masked", async () => {
    expect(
      await veil.mask({
        text: "Zeynep",
        expectName: true,
        never: ["Zeynep"],
        always: ["Zeynep"]
      })
    ).toBe("[NAME]")
  })

  it("expectName: leading spaces do not pass the size limit", async () => {
    await expect(
      veil.mask({ text: `${" ".repeat(20001)}Zeynep`, expectName: true })
    ).rejects.toThrow("longer than")
  })

  it("expectName: a long answer still meets the size limit", async () => {
    await expect(
      veil.mask({ text: "A".repeat(20001), expectName: true })
    ).rejects.toThrow("longer than")
  })

  it("expectName: leaves refusals, digits, emails and paths", async () => {
    for (const text of ["skip", "12345", "a@b.co", "why?", "C:\\Anna"]) {
      expect(await veil.mask({ text, expectName: true })).toBe(text)
    }
  })

  it("threshold: a lower value masks a doubtful name", async () => {
    const options = { text: "River", context: "What is your full name?" }
    expect(await veil.mask({ ...options, threshold: 0.5 })).toBe("River")
    expect(await veil.mask({ ...options, threshold: 0.05 })).toBe("[NAME]")
  })

  it("threshold: rejects a value outside 0 to 1", async () => {
    await expect(veil.detect({ text: "hi", threshold: 0 })).rejects.toThrow(
      "threshold"
    )
  })

  it("types: finds the listed kinds and names each replacement", async () => {
    const text = "I am Anna Schmidt, born 3 May 1990, call +49 151 23456789"
    expect(await veil.mask({ text })).toBe(
      "I am [NAME], born 3 May 1990, call +49 151 23456789"
    )
    expect(await veil.mask({ text, types: ["DOB", "PHONE"] })).toBe(
      "I am Anna Schmidt, born [DATE OF BIRTH], call [PHONE]"
    )
  })

  it.each([
    [
      "Please send it to 42 Baker Street, Flat 3B, Leeds LS1 4AP.",
      "Please send it to [ADDRESS]."
    ],
    [
      "My passport number is K4829175 and my customer ID is CU-558213.",
      "My passport number is [ID] and my customer ID is [ID]."
    ],
    [
      "Call me on +44 7700 900123 or mail sam.okafor@example.com.",
      "Call me on [PHONE] or mail [EMAIL]."
    ],
    [
      "The card 4111 1111 1111 1111 was charged twice, please refund.",
      "The card [FINANCIAL] was charged twice, please refund."
    ],
    [
      "Ich heiße Jonas Becker, meine Adresse ist Lindenstraße 17, 50674 Köln.",
      "Ich heiße [NAME], meine Adresse ist [ADDRESS]."
    ],
    [
      "Can I change my plan from Basic to Premium before Monday?",
      "Can I change my plan from Basic to Premium before Monday?"
    ]
  ])("types: masks every kind in %s", async (text, masked) => {
    expect(await veil.mask({ text, types: [...SPAN_TYPES] })).toBe(masked)
  })

  it("returns no span for a text without a Latin letter", async () => {
    for (const text of ["", "12345", "👍", "Ελένη Παπαδοπούλου"]) {
      expect(await veil.detect({ text, context: "Your name?" })).toEqual([])
    }
  })

  it("does not mask inside an email address or a link", async () => {
    const text =
      "write to anna.schmidt@example.com or see https://example.com/anna-schmidt"
    expect(await veil.mask({ text })).toBe(text)
  })

  it("refuses a text over the size limit", async () => {
    await expect(veil.mask({ text: "a ".repeat(15000) })).rejects.toThrow(
      "longer than"
    )
  })
})
