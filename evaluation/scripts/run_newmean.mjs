import { readFileSync, writeFileSync } from "node:fs"
import { createRedactor } from "@newmean-ai/pii"
const [size, input, output] = process.argv.slice(2)
const redactor = await createRedactor({ size })
const lines = readFileSync(input, "utf8").trim().split("\n")
const out = []
const t0 = performance.now()
for (const line of lines) {
  const row = JSON.parse(line)
  const spans = await redactor.detect(row.text)
  out.push(JSON.stringify({ id: row.id, pred: spans.filter((s) => s.label === "PERSON" || s.label === "PERSON_L2").map((s) => [s.start, s.end]) }))
}
writeFileSync(output, out.join("\n") + "\n")
console.log(size, input.split("/").pop(), lines.length, "rows", Math.round(performance.now() - t0), "ms", JSON.stringify(redactor.info?.modelVersion))
