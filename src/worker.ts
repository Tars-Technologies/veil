/// <reference lib="webworker" />
import type { MaskOptions } from "./detect"
import { localRuntime } from "./veil"

export type Request = {
  id: number
  model?: ArrayBuffer | string
  options?: MaskOptions
}

export type Reply = { id: number; result?: unknown; error?: string }

let runtime: ReturnType<typeof localRuntime> | undefined

self.onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    if (data.model !== undefined) runtime = localRuntime(data.model)
    if (!runtime) throw new Error("veil: the worker has no model")
    const loaded = await runtime
    const reply: Reply = {
      id: data.id,
      result: data.options ? await loaded.detect(data.options) : loaded.model
    }
    self.postMessage(reply)
  } catch (error) {
    const reply: Reply = {
      id: data.id,
      error: error instanceof Error ? error.message : String(error)
    }
    self.postMessage(reply)
  }
}
