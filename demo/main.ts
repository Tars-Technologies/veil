import modelUrl from "../veil-0.3.1.bin?url"
import { createVeil, type Span, type SpanType } from "../src/browser"

const TYPES: { type: SpanType; name: string; mask: string }[] = [
  { type: "PERSON", name: "Name", mask: "[NAME]" },
  { type: "ADDRESS", name: "Address", mask: "[ADDRESS]" },
  { type: "DOB", name: "Date of birth", mask: "[DATE OF BIRTH]" },
  { type: "ID", name: "ID number", mask: "[ID]" },
  { type: "FINANCIAL", name: "Financial", mask: "[FINANCIAL]" },
  { type: "PHONE", name: "Phone", mask: "[PHONE]" },
  { type: "EMAIL", name: "Email", mask: "[EMAIL]" }
]

const SAMPLES: { name: string; text: string }[] = [
  {
    name: "Support chat",
    text:
      "Hi, this is Laura Bianchi. My parcel has not arrived yet. Please send it to 14 Rue des Lilas, 69003 Lyon, " +
      "or call me on +33 6 39 98 12 34. My email is laura.bianchi@example.com."
  },
  {
    name: "Account form",
    text:
      "Name: Daniel Okafor\nDate of birth: 03/07/1988\nPassport: K4829175\nIBAN: DE89 3704 0044 0532 0130 00\n" +
      "Address: 221 Baker Street, London NW1 6XE"
  },
  {
    name: "Team message",
    text:
      "Thanks Priya! I spoke with Mr. Jonas Weber this morning and he will send the contract to Sofia Marquez " +
      "before Friday."
  },
  {
    name: "Three languages",
    text:
      "Bonjour, je m'appelle Camille Durand. Mein Kollege Lukas Hoffmann hat die Rechnung geschickt. " +
      "Mi compañera Lucía Fernández llamará mañana."
  }
]

const find = <T extends HTMLElement>(id: string) => document.getElementById(id) as T | null
const text = find<HTMLTextAreaElement>("text")!
const highlighted = find<HTMLDivElement>("highlighted")
const masked = find<HTMLDivElement>("masked")
const json = find<HTMLPreElement>("json")
const threshold = find<HTMLInputElement>("threshold")!
const thresholdValue = find<HTMLOutputElement>("threshold-value")!
const status = find<HTMLElement>("status")!
const timing = find<HTMLElement>("timing")
const copy = find<HTMLButtonElement>("copy")

const selected = new Set<SpanType>(TYPES.map(({ type }) => type))
const counts = new Map<SpanType, HTMLElement>()
for (const { type, name } of TYPES) {
  const button = document.createElement("button")
  button.type = "button"
  button.className = `chip ${type}`
  button.setAttribute("aria-pressed", "true")
  const dot = document.createElement("span")
  dot.className = "dot"
  const count = document.createElement("span")
  count.className = "count"
  counts.set(type, count)
  button.append(dot, name, count)
  button.addEventListener("click", () => {
    const on = !selected.has(type)
    if (on) selected.add(type)
    else selected.delete(type)
    button.setAttribute("aria-pressed", String(on))
    update()
  })
  find("types")!.append(button)
}

const sampleButtons = SAMPLES.map((sample, index) => {
  const button = document.createElement("button")
  button.type = "button"
  button.className = "sample"
  button.textContent = sample.name
  button.setAttribute("aria-pressed", String(index === 0))
  button.addEventListener("click", () => {
    text.value = sample.text
    update()
  })
  find("samples")!.append(button)
  return button
})

for (const tab of document.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll<HTMLButtonElement>("[data-tab]"))
      other.setAttribute("aria-selected", String(other === tab))
    for (const panel of document.querySelectorAll<HTMLElement>("[data-panel]"))
      panel.hidden = panel.dataset.panel !== tab.dataset.tab
  })
}

const maskOf = (type: SpanType) => TYPES.find((entry) => entry.type === type)!.mask

function render(target: HTMLElement, value: string, spans: Span[], mode: "label" | "mask") {
  target.replaceChildren()
  let at = 0
  for (const span of spans) {
    target.append(value.slice(at, span.start))
    const mark = document.createElement("mark")
    mark.className = span.type
    if (mode === "mask") mark.textContent = maskOf(span.type)
    else mark.textContent = value.slice(span.start, span.end)
    if (mode === "label") mark.dataset.label = TYPES.find((entry) => entry.type === span.type)!.name.toLowerCase()
    target.append(mark)
    at = span.end
  }
  target.append(value.slice(at))
}

const veil = createVeil({ model: modelUrl })
let request = 0

async function update() {
  const value = text.value
  thresholdValue.value = Number(threshold.value).toFixed(2)
  const current = ++request
  const started = performance.now()
  let spans: Span[]
  try {
    spans = await veil.detect({
      text: value,
      threshold: Number(threshold.value),
      types: TYPES.map(({ type }) => type).filter((type) => selected.has(type))
    })
  } catch (error) {
    if (current === request) status.textContent = (error as Error).message.replace(/^veil: /, "")
    return
  }
  if (current !== request) return
  const elapsed = performance.now() - started
  if (highlighted) render(highlighted, value, spans, "label")
  if (masked) render(masked, value, spans, "mask")
  if (json)
    json.textContent = JSON.stringify(
      spans.map((span) => ({ ...span, text: value.slice(span.start, span.end) })),
      null,
      2
    )
  for (const [type, count] of counts) {
    const found = spans.filter((span) => span.type === type).length
    count.textContent = found ? String(found) : ""
  }
  status.textContent = `${spans.length} found`
  for (const [index, button] of sampleButtons.entries())
    button.setAttribute("aria-pressed", String(SAMPLES[index].text === value))
  if (timing) timing.textContent = `${elapsed.toFixed(0)} ms`
}

copy?.addEventListener("click", async () => {
  const shown = document.querySelector<HTMLElement>("[data-panel]:not([hidden])")
  await navigator.clipboard.writeText(shown?.textContent ?? "")
  copy.textContent = "Copied"
  setTimeout(() => (copy.textContent = "Copy"), 1200)
})

text.addEventListener("input", update)
threshold.addEventListener("input", update)
text.value = SAMPLES[0].text
veil.ready().then(update).catch((error: Error) => {
  status.textContent = `The model failed to load: ${error.message}`
})
