import React, { useState, useEffect, useRef, useCallback } from "react";
import { useVerification } from "../context/verification";
import { registerEmployee, validatePose } from "../services/apiClient";

// 3 poses matching Python GUI
const POSES: Array<{ id: "center" | "left" | "right"; label: string; instruction: string }> = [
    { id: "center", label: "Face Forward", instruction: "Look directly at the camera" },
    { id: "left", label: "Turn Right", instruction: "Turn your head slightly to the right" },
    { id: "right", label: "Turn Left", instruction: "Turn your head slightly to the left" },
];

type RegistrationStep = "initial" | "instructions" | "capture" | "review" | "completing";

export const RegistrationWizard = () => {
    const { capture, isReady, stream } = useVerification();
    const [step, setStep] = useState<RegistrationStep>("initial");
    const [name, setName] = useState("");
    const [currentPoseIndex, setCurrentPoseIndex] = useState(0);
    const [captures, setCaptures] = useState<string[]>([]);
    const [status, setStatus] = useState<string>("");
    const [error, setError] = useState<string | null>(null);

    // Local video ref for registration preview
    const localVideoRef = useRef<HTMLVideoElement | null>(null);

    // Pose validation state
    const [poseValid, setPoseValid] = useState(false);
    const [poseFeedback, setPoseFeedback] = useState("");
    const validationIntervalRef = useRef<number | null>(null);

    // Locked frame state for auto-capture
    const [lockedFrame, setLockedFrame] = useState<string | null>(null);

    // Assign stream to local video when entering capture step OR when unlocking frame
    useEffect(() => {
        if (step === "capture" && !lockedFrame && localVideoRef.current && stream) {
            localVideoRef.current.srcObject = stream;
        }
    }, [step, stream, lockedFrame]);

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
        setPoseValid(false);
        setPoseFeedback("Positioning...");
    };



    // Real-time pose validation loop
    const validateCurrentPose = useCallback(async () => {
        if (step !== "capture" || !isReady || lockedFrame) return;

        try {
            const image = await capture();
            const targetPose = POSES[currentPoseIndex].id;
            const result = await validatePose(image, targetPose);

            if (result.valid) {
                // Auto-capture on success
                // Use the cropped face image from backend if available, otherwise fallback to full frame
                setLockedFrame(result.face_image || image);
                setPoseValid(true);
                setPoseFeedback("Captured! Please confirm.");

                // Stop validation loop temporarily
                if (validationIntervalRef.current) {
                    clearInterval(validationIntervalRef.current);
                    validationIntervalRef.current = null;
                }
            } else {
                setPoseValid(false);
                setPoseFeedback(result.feedback);
            }
        } catch (err) {
            setPoseFeedback("Checking pose...");
        }
    }, [step, isReady, capture, currentPoseIndex, lockedFrame]);

    // Start/stop pose validation interval
    useEffect(() => {
        if (step === "capture" && !lockedFrame) {
            // Validate every 200ms (faster for responsiveness)
            validationIntervalRef.current = window.setInterval(validateCurrentPose, 200);
        } else {
            if (validationIntervalRef.current) {
                clearInterval(validationIntervalRef.current);
                validationIntervalRef.current = null;
            }
        }

        return () => {
            if (validationIntervalRef.current) {
                clearInterval(validationIntervalRef.current);
            }
        };
    }, [step, validateCurrentPose, lockedFrame]);

    const confirmCapture = () => {
        if (!lockedFrame) return;

        const newCaptures = [...captures, lockedFrame];
        setCaptures(newCaptures);
        setLockedFrame(null); // Reset lock

        if (currentPoseIndex < POSES.length - 1) {
            setCurrentPoseIndex((prev) => prev + 1);
            setPoseValid(false);
            setPoseFeedback("Positioning...");
        } else {
            setStep("review");
        }
    };

    const retakeCapture = () => {
        setLockedFrame(null);
        setPoseValid(false);
        setPoseFeedback("Positioning...");
        // Effect will restart validation since lockedFrame is null
    };

    const handleSubmit = async () => {
        setStep("completing");
        setStatus("Registering...");
        try {
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

    const cancelRegistration = () => {
        setStep("initial");
        setName("");
        setCaptures([]);
        setCurrentPoseIndex(0);
        setPoseValid(false);
        setPoseFeedback("");
        setError(null);
        setLockedFrame(null);
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
                    <p>• You will capture 3 poses: <strong>Center</strong>, <strong>Left</strong>, and <strong>Right</strong>.</p>
                    <p>• Wait for the green "✓ Hold steady!" before capturing.</p>
                </div>
                <div className="wizard-actions">
                    <button onClick={cancelRegistration} className="action-btn secondary">
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

                {/* Live Camera Preview or Locked Frame */}
                <div className="registration-preview" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    {lockedFrame ? (
                        <img
                            src={lockedFrame}
                            alt="Captured Pose"
                            className="preview-video mirrored"
                            style={{ objectFit: 'contain', maxHeight: '100%', width: 'auto' }}
                        />
                    ) : (
                        <video
                            ref={localVideoRef}
                            autoPlay
                            playsInline
                            muted
                            className="preview-video mirrored"
                        />
                    )}
                    <div className={`pose-overlay ${poseValid ? "valid" : "invalid"}`}>
                        {poseFeedback}
                    </div>
                </div>

                {lockedFrame ? (
                    <div className="wizard-actions full-width">
                        <button onClick={confirmCapture} className="action-btn primary large">
                            Confirm & Continue
                        </button>
                        <button onClick={retakeCapture} className="action-btn secondary large">
                            Retake
                        </button>
                    </div>
                ) : (
                    <button
                        disabled
                        className="action-btn secondary full-width large"
                    >
                        Adjust your pose...
                    </button>
                )}

                <button onClick={cancelRegistration} className="link-btn">
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
                            <span className="thumbnail-label">{POSES[i]?.label}</span>
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
