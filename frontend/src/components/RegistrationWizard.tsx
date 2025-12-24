import React, { useState } from "react";
import { useVerification } from "../context/verification";
import { registerEmployee } from "../services/apiClient";
// Internal constants
const REGISTRATION_STATES = {
    idle: "idle",
    submitting: "submitting",
    success: "success",
    error: "error",
} as const;


type RegistrationStep = "initial" | "instructions" | "capture" | "review" | "completing";

const POSES = [
    { id: "center", label: "Face Forward", instruction: "Look directly at the camera" },
    { id: "left", label: "Turn Left", instruction: "Turn your head slightly to the left" },
    { id: "right", label: "Turn Right", instruction: "Turn your head slightly to the right" },
    { id: "up", label: "Look Up", instruction: "Tilt your head slightly up" },
    { id: "down", label: "Look Down", instruction: "Tilt your head slightly down" },
];

export const RegistrationWizard = () => {
    const { capture, isReady } = useVerification();
    const [step, setStep] = useState<RegistrationStep>("initial");
    const [name, setName] = useState("");
    const [currentPoseIndex, setCurrentPoseIndex] = useState(0);
    const [captures, setCaptures] = useState<string[]>([]);
    const [status, setStatus] = useState<string>("");
    const [error, setError] = useState<string | null>(null);

    const startRegistration = () => {
        if (!name.trim()) {
            setError("Please enter a name");
            return;
        }
        setError(null);
        setStep("instructions");
    };

    const startCapture = () => {
        setStep("capture");
        setCurrentPoseIndex(0);
        setCaptures([]);
    };

    const handleCapture = async () => {
        try {
            const image = await capture();
            const newCaptures = [...captures, image];
            setCaptures(newCaptures);

            if (currentPoseIndex < POSES.length - 1) {
                setCurrentPoseIndex((prev) => prev + 1);
            } else {
                setStep("review");
            }
        } catch (err) {
            setError("Failed to capture image");
        }
    };

    const handleSubmit = async () => {
        setStep("completing");
        setStatus("Registering...");
        try {
            console.log("Submitting registration:", { name, images: captures.length });
            await registerEmployee({
                name,
                images_b64: captures,
            });
            setStatus("Registration Successful!");
            setTimeout(() => {
                setStep("initial");
                setName("");
                setCaptures([]);
                setStatus("");
            }, 2000);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Registration failed");
            setStep("review");
        }
    };

    if (step === "initial") {
        return (
            <div className="wizard-step">
                <div className="field-group">
                    <label>Employee Name</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Enter name"
                        autoFocus
                    />
                    {error && <p className="error">{error}</p>}
                </div>
                <button onClick={startRegistration} className="action-btn primary full-width">
                    Start Registration
                </button>
            </div>
        );
    }

    if (step === "instructions") {
        return (
            <div className="wizard-step">
                <div className="info-box">
                    <p>• Position yourself 2-3 feet from the camera.</p>
                    <p>• Ensure good lighting on your face.</p>
                    <p>• You will be asked to capture 3 poses: Center, Left, and Right.</p>
                </div>
                <div className="wizard-actions">
                    <button onClick={() => setStep("initial")} className="action-btn secondary">
                        Cancel
                    </button>
                    <button onClick={startCapture} disabled={!isReady} className="action-btn primary">
                        Begin Capture
                    </button>
                </div>
            </div>
        );
    }

    if (step === "capture") {
        const currentPose = POSES[currentPoseIndex];
        return (
            <div className="wizard-step">
                <div className="step-header">
                    <h4>Step {currentPoseIndex + 1}/{POSES.length}</h4>
                    <h3>{currentPose.label}</h3>
                    <p className="subtitle">{currentPose.instruction}</p>
                </div>

                <button onClick={handleCapture} className="action-btn primary full-width large">
                    Capture Frame
                </button>
                <button onClick={() => setStep("initial")} className="link-btn">
                    Cancel
                </button>
            </div>
        );
    }

    if (step === "review") {
        return (
            <div className="wizard-step">
                <div className="thumbnails-grid">
                    {captures.map((img, i) => (
                        <div key={i} className="thumbnail">
                            <img src={img} alt={`Pose ${i}`} />
                        </div>
                    ))}
                </div>
                {error && <p className="error">{error}</p>}
                <div className="wizard-actions">
                    <button onClick={startCapture} className="action-btn secondary">
                        Retake
                    </button>
                    <button onClick={handleSubmit} className="action-btn primary">
                        Submit Registration
                    </button>
                </div>
            </div>
        );
    }

    if (step === "completing") {
        return (
            <div className="wizard-step centered">
                <h3>{status || "Processing..."}</h3>
                <div className="loading-bar"></div>
            </div>
        );
    }

    return null;
};
