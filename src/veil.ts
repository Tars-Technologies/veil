// Runs the model on the calling thread: servers, serverless functions, tests. In a browser, import "@tars-technologies/veil/browser".
import {
  createClient,
  type Runtime,
  type Veil,
  type VeilOptions
} from "./client.ts"
import { detect } from "./detect.ts"
import { Engine } from "./engine.ts"

export {
  type Defaults,
  type MaskOptions,
  type ModelInfo,
  type Span,
  type SpanType,
  type Veil,
  type VeilOptions,
  version
} from "./client.ts"

export async function localRuntime(
  model: ArrayBuffer | string
): Promise<Runtime> {
  const engine = new Engine(
    typeof model === "string" ? await download(model) : model
  )
  return { model: engine.model, detect: (options) => detect(engine, options) }
}

async function download(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url)
  if (!response.ok)
    throw new Error(
      `veil: model download failed with status ${response.status}`
    )
  return response.arrayBuffer()
}

export const createVeil = (options: VeilOptions): Veil =>
  createClient(localRuntime, options)
