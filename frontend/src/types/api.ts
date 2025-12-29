export interface VerifyRequest {
  image_b64: string; // base64 or data URL from webcam capture
  threshold?: number;
  mark_attendance?: boolean;
  blink_sequence?: string[];
}

export interface BlinkDetails {
  has_blinked: boolean;
  blink_score: number;
  frames_processed: number;
  min_ear?: number;
  blinks_needed: number;
}

export interface VerifyResponse {
  identity: string | null;
  distance: number;
  confidence: number;
  // Backend returns "Real" | "Spoof"
  liveness: string;
  threshold: number;
  metadata?: Record<string, unknown>;
  detections?: DetectedFace[];
  blink?: BlinkDetails;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedFace {
  bbox: BoundingBox;
  is_primary: boolean;
  identity?: string | null;
  confidence?: number;
  liveness?: string;
}

export interface RegisterRequest {
  name: string;
  images_b64: string[];
  replace_existing?: boolean;
}

export interface RegisterResponse {
  success: boolean;
  name: string;
  poses: number;
}

export interface ThresholdPayload {
  threshold: number;
}

// Matches AttendanceRecordRequest in backend
export interface AttendanceEvent {
  name: string;
  distance: number;
  emotion?: string;
  liveness?: string;
}

// Matches /employees response
export interface EmployeeSummary {
  count: number;
  employees: string[];
}

export interface DailyAttendanceRecord {
  timestamp: string;
  employee_name: string;
  prob_distance: number; // or string based on backend serialization? backend sends "distance" in payload but in summary it might be different. 
  // checking backend: _serialize_dataframe sends dataframe records. columns likely match logger.
  // let's stick to generic keys for now or check backend logger.
  // View backend logger to be sure.
  [key: string]: unknown;
}

export interface AttendanceSummary {
  total_count: number;
  unique_employees: number;
  avg_confidence: number;
  liveness_real_count: number;
  liveness_spoof_count: number;
}

// Pose validation types
export interface PoseValidateRequest {
  image_b64: string;
  target_pose: "center" | "left" | "right";
}

export interface PoseValidateResponse {
  valid: boolean;
  detected_pose: string;
  target_pose: string;
  feedback: string;
  yaw: number;
  pitch: number;
  face_image?: string;
}

export interface DeleteEmployeeResponse {
  success: boolean;
  name: string;
  remaining: number;
}
