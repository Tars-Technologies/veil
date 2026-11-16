import packageJson from "../package.json"
import { type MaskOptions, redact } from "./detect"
import type { ModelInfo, Span, SpanType } from "./engine"

export type { MaskOptions, ModelInfo, Span, SpanType }

export const version: string = packageJson.version

export type Defaults = Omit<MaskOptions, "text">

export type VeilOptions = {
  /** The bytes of the model file, or a URL to it. */
  model: ArrayBuffer | string
  /** Options applied to every call; a call can override each of them. */
  defaults?: Defaults
}

export type Veil = {
  /** "idle" until the first call or `ready()`. */
  readonly status: "idle" | "loading" | "ready" | "error"
  /** Name, version and format of the loaded model; null until ready. */
  readonly model: ModelInfo | null
  /** The defaults as given to `createVeil`. They cannot change; create a new client for other defaults. */
  readonly defaults: Readonly<Defaults>
  /** Loads the model. Optional: `detect` and `mask` load it on first use. */
  ready(): Promise<void>
  /** Spans of personal data in `text`, as UTF-16 offsets, in order, without overlap. */
  detect(options: MaskOptions): Promise<Span[]>
  /** `text` with every span replaced. */
  mask(options: MaskOptions): Promise<string>
}

/** A loaded model, on this thread or behind a worker. */
export type Runtime = {
  model: ModelInfo
  detect(options: MaskOptions): Span[] | Promise<Span[]>
}

export function createClient(
  start: (
    model: ArrayBuffer | string,
    fail: (error: Error) => void
  ) => Promise<Runtime>,
  options: VeilOptions
): Veil {
  const { model } = options
  // A copy that cannot change: `mask`, `detect` and the React hook then always agree on the defaults.
  const defaults: Readonly<Defaults> = Object.freeze({
    ...options.defaults,
    always: Object.freeze([...(options.defaults?.always ?? [])]) as string[],
    never: Object.freeze([...(options.defaults?.never ?? [])]) as string[],
    types:
      options.defaults?.types &&
      (Object.freeze([...options.defaults.types]) as SpanType[])
  })
  let status: Veil["status"] = "idle"
  let info: ModelInfo | null = null
  let runtime: Promise<Runtime> | null = null

  // A runtime that breaks after loading: every later call rejects with its error.
  const fail = (error: Error) => {
    status = "error"
    runtime = Promise.reject(error)
    runtime.catch(() => {})
  }
  const load = () => {
    if (!runtime) {
      status = "loading"
      runtime = start(model, fail).then(
        (loaded) => {
          status = "ready"
          info = loaded.model
          return loaded
        },
        (error) => {
          status = "error"
          throw error
        }
      )
    }
    return runtime
  }
  // The options are resolved and copied before the first wait, so a caller that changes its object or its lists
  // afterwards cannot change the result. A default applies when the call gives no value.
  const run = async (options: MaskOptions) => {
    const types = options.types ?? defaults.types
    const call: MaskOptions = {
      text: options.text,
      context: options.context ?? defaults.context,
      threshold: options.threshold ?? defaults.threshold,
      types: types && [...types],
      always: [...(options.always ?? defaults.always ?? [])],
      never: [...(options.never ?? defaults.never ?? [])],
      expectName: options.expectName ?? defaults.expectName,
      replacement: options.replacement ?? defaults.replacement
    }
    const spans = await (await load()).detect(call)
    return { call, spans }
  }

  return {
    get status() {
      return status
    },
    get model() {
      return info
    },
    defaults,
    ready: async () => void (await load()),
    detect: async (options) => (await run(options)).spans,
    mask: async (options) => {
      const { call, spans } = await run(options)
      return redact(call.text, spans, call.replacement)
    }
  }
}
