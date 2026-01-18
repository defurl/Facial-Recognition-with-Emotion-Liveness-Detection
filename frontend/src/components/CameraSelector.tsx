import { useVerification } from "../context/verification";

export function CameraSelector() {
  const { cameras, selectedCamera, switchCamera, refreshCameras } = useVerification();

  const handleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const deviceId = e.target.value;
    if (deviceId) {
      await switchCamera(deviceId);
    }
  };

  return (
    <div className="camera-selector">
      <div className="field-row">
        <label>Camera</label>
        <button className="ghost-button" onClick={refreshCameras} title="Refresh camera list">
          ↻
        </button>
      </div>
      <select
        className="compact-input"
        value={selectedCamera || ""}
        onChange={handleChange}
        style={{ width: "100%", marginTop: "0.5rem" }}
      >
        {cameras.length === 0 ? (
          <option value="">No cameras found</option>
        ) : (
          cameras.map((cam) => (
            <option key={cam.deviceId} value={cam.deviceId}>
              {cam.label}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
