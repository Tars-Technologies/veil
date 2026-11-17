import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useEffect, useState } from "react";
import { redact } from "./detect.js";
const VeilContext = createContext(null);
export function VeilProvider({ veil, children }) {
    return _jsx(VeilContext.Provider, { value: veil, children: children });
}
export function useVeil() {
    const veil = useContext(VeilContext);
    if (!veil)
        throw new Error("veil: useVeil needs a <VeilProvider>");
    return veil;
}
/** Masks `options.text` again each time the options change. For a live preview; to mask on send, call `useVeil().mask`. */
export function useMask(options) {
    const veil = useVeil();
    const key = JSON.stringify(options);
    const [settled, setSettled] = useState({});
    // `key` stands for `options`, which is a new object on every render.
    useEffect(() => {
        let current = true;
        veil.detect(options).then((spans) => current && setSettled({ veil, key, spans }), (error) => current && setSettled({ veil, key, error }));
        return () => {
            current = false;
        };
    }, [veil, key]);
    const { spans, error = null } = settled.veil === veil && settled.key === key ? settled : {};
    return {
        data: spans &&
            redact(options.text, spans, options.replacement ?? veil.defaults.replacement),
        spans,
        status: spans ? "success" : error ? "error" : "pending",
        isPending: !spans && !error,
        error
    };
}
