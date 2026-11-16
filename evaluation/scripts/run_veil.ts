import { readFileSync, writeFileSync } from "node:fs"
import { detectChunked } from "../../src/chunked"
import { loadModel } from "../../src/model"
import { createVeil } from "../../src/veil"

const [input, output] = process.argv.slice(2)
const veil = createVeil({ model: loadModel() })
const lines = readFileSync(input, "utf8").trim().split("\n")
const out: string[] = []
const started = performance.now()
for (const line of lines) {
  const row = JSON.parse(line)
  const spans = await detectChunked(veil, row.text)
  out.push(JSON.stringify({ id: row.id, pred: spans.filter((s) => s.type === "PERSON").map((s) => [s.start, s.end]) }))
}
writeFileSync(output, `${out.join("\n")}\n`)
console.log(input.split("/").pop(), lines.length, "rows", Math.round(performance.now() - started), "ms")
