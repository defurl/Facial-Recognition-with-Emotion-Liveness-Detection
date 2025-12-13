import { FormEvent, useMemo, useState } from "react";
import { registerEmployee } from "../services/apiClient";
import { useVerification } from "../context/verification";

const REGISTRATION_STATES = {
  idle: "idle",
  submitting: "submitting",
  success: "success",
  error: "error",
} as const;

type RegistrationState = (typeof REGISTRATION_STATES)[keyof typeof REGISTRATION_STATES];

type RegistrationResult = {
  status: RegistrationState;
  message?: string;
};

export const RegistrationPanel = () => {
  const { lastCapture, lastResult } = useVerification();
  const [name, setName] = useState("Zilus");
  const [manualBase64, setManualBase64] = useState("");
  const [replaceExisting, setReplaceExisting] = useState(false);
  const [result, setResult] = useState<RegistrationResult>({ status: REGISTRATION_STATES.idle });

  const snapshotForSubmission = useMemo(() => {
    if (lastCapture) {
      return lastCapture;
    }
    const trimmed = manualBase64.trim();
    return trimmed ? trimmed : undefined;
  }, [manualBase64, lastCapture]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!snapshotForSubmission) {
      setResult({ status: REGISTRATION_STATES.error, message: "Provide a capture or enter base64 to register." });
      return;
    }

    setResult({ status: REGISTRATION_STATES.submitting });

    try {
      const response = await registerEmployee({
        name,
        images_b64: [snapshotForSubmission],
        replace_existing: replaceExisting,
      });

      setResult({
        status: REGISTRATION_STATES.success,
        message: `Registered ${response.name} with ${response.poses} capture${response.poses === 1 ? "" : "s"}`,
      });
    } catch (error) {
      setResult({ status: REGISTRATION_STATES.error, message: error instanceof Error ? error.message : "Unexpected error" });
    }
  };

  return (
    <section className="panel">
      <header>
        <h3>Registration</h3>
        <p className="hint">
          {lastResult
            ? "Using the capture from the latest successful verification"
            : "Drop an image as base64 if you want to prefill registration without verifying"
          }
        </p>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="field-group">
          <label htmlFor="name">Name</label>
          <input id="name" value={name} onChange={(event) => setName(event.target.value)} required />
        </div>

        <div className="field-group">
          <label htmlFor="base64">Base64 Image (optional)</label>
          <textarea
            id="base64"
            value={manualBase64}
            onChange={(event) => setManualBase64(event.target.value)}
            placeholder="Paste a face capture if you do not want to rely on the last verification capture"
          />
        </div>

        <div className="field-group inline">
          <label htmlFor="replace">
            <input
              id="replace"
              type="checkbox"
              checked={replaceExisting}
              onChange={(event) => setReplaceExisting(event.target.checked)}
            />
            Replace existing entry if the name already exists
          </label>
        </div>

        <button type="submit" disabled={result.status === REGISTRATION_STATES.submitting}>
          {result.status === REGISTRATION_STATES.submitting ? "Registering..." : "Register employee"}
        </button>
      </form>

      {result.status === REGISTRATION_STATES.error && <p className="error">{result.message}</p>}
      {result.status === REGISTRATION_STATES.success && <p className="success">{result.message}</p>}
    </section>
  );
};
