import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useState
} from "react"
import type { Veil } from "./client.ts"
import { type MaskOptions, redact } from "./detect.ts"
import type { Span } from "./engine.ts"

const VeilContext = createContext<Veil | null>(null)

export function VeilProvider({
  veil,
  children
}: {
  veil: Veil
  children: ReactNode
}) {
  return <VeilContext.Provider value={veil}>{children}</VeilContext.Provider>
}

export function useVeil(): Veil {
  const veil = useContext(VeilContext)
  if (!veil) throw new Error("veil: useVeil needs a <VeilProvider>")
  return veil
}

export type MaskResult = {
  /** The masked text; undefined while pending or on error. */
  data: string | undefined
  spans: Span[] | undefined
  status: "pending" | "success" | "error"
  isPending: boolean
  error: Error | null
}

/** Masks `options.text` again each time the options change. For a live preview; to mask on send, call `useVeil().mask`. */
export function useMask(options: MaskOptions): MaskResult {
  const veil = useVeil()
  const key = JSON.stringify(options)
  const [settled, setSettled] = useState<{
    veil?: Veil
    key?: string
    spans?: Span[]
    error?: Error
  }>({})

  // `key` stands for `options`, which is a new object on every render.
  useEffect(() => {
    let current = true
    veil.detect(options).then(
      (spans) => current && setSettled({ veil, key, spans }),
      (error: Error) => current && setSettled({ veil, key, error })
    )
    return () => {
      current = false
    }
  }, [veil, key])

  const { spans, error = null } =
    settled.veil === veil && settled.key === key ? settled : {}
  return {
    data:
      spans &&
      redact(
        options.text,
        spans,
        options.replacement ?? veil.defaults.replacement
      ),
    spans,
    status: spans ? "success" : error ? "error" : "pending",
    isPending: !spans && !error,
    error
  }
}
