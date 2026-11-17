import fs from "node:fs"

const file = "dist/browser.js"
const source = fs.readFileSync(file, "utf8")
if (!source.includes('"./worker.ts"')) throw new Error(`${file} has no "./worker.ts" to rewrite`)
fs.writeFileSync(file, source.replace('"./worker.ts"', '"./worker.js"'))
