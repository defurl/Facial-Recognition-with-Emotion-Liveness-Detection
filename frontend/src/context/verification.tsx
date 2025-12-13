import { createContext, ReactNode, useContext, useMemo, useState } from "react";
import type { VerifyResponse } from "../types/api";

interface VerificationContextType {
  lastCapture: string | null;
  lastResult: VerifyResponse | null;
  setLastCapture: (value: string | null) => void;
  setLastResult: (value: VerifyResponse | null) => void;
}

const VerificationContext = createContext<VerificationContextType | undefined>(undefined);

export function VerificationProvider({ children }: { children: ReactNode }) {
  const [lastCapture, setLastCapture] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<VerifyResponse | null>(null);

  const value = useMemo(
    () => ({ lastCapture, lastResult, setLastCapture, setLastResult }),
    [lastCapture, lastResult]
  );

  return <VerificationContext.Provider value={value}>{children}</VerificationContext.Provider>;
}

export function useVerification() {
  const context = useContext(VerificationContext);
  if (!context) {
    throw new Error("useVerification must be used within VerificationProvider");
  }
  return context;
}
