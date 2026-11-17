// The same API as "@tars-technologies/veil", with the model in a Web Worker, so the page never waits for it.
import {
  createClient,
  type ModelInfo,
  type Runtime,
  type Span,
  type Veil,
  type VeilOptions
} from "./client.ts"
import type { Reply, Request } from "./worker.ts"

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

async function workerRuntime(
  model: ArrayBuffer | string,
  fail: (error: Error) => void
): Promise<Runtime> {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module"
  })
  const waiting = new Map<
    number,
    { resolve: (result: unknown) => void; reject: (error: Error) => void }
  >()
  let next = 0
  worker.onmessage = ({ data }: MessageEvent<Reply>) => {
    const entry = waiting.get(data.id)
    waiting.delete(data.id)
    if (data.error === undefined) entry?.resolve(data.result)
    else entry?.reject(new Error(data.error))
  }
  // A worker that fails to load or throws outside a request would leave every call waiting, now and later.
  worker.onerror = (event) => {
    const error = new Error(`veil: the worker failed: ${event.message}`)
    for (const entry of waiting.values()) entry.reject(error)
    waiting.clear()
    worker.terminate()
    fail(error)
  }
  const call = <Result>(message: Omit<Request, "id">) =>
    new Promise<Result>((resolve, reject) => {
      waiting.set(next, {
        resolve: resolve as (result: unknown) => void,
        reject
      })
      const request: Request = { id: next++, ...message }
      worker.postMessage(request)
    })
  // A relative URL must resolve against the page; inside the worker it would resolve against the worker script.
  const source =
    typeof model === "string" ? new URL(model, document.baseURI).href : model
  return {
    model: await call<ModelInfo>({ model: source }),
    detect: (options) => call<Span[]>({ options })
  }
}

export const createVeil = (options: VeilOptions): Veil =>
  createClient(workerRuntime, options)
