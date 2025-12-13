export interface VerifyRequest {
  image?: string; // base64 or data URL from webcam capture
  embedding?: number[]; // optional precomputed embedding
}

export interface VerifyResponse {
  identity: string | null;
  distance: number;
  confidence: number;
  liveness: "pass" | "fail" | "unknown";
  metadata?: Record<string, unknown>;
  detections?: DetectedFace[];
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

export interface AttendanceEvent {
  identity: string;
  status: string;
  timestamp: string;
}

export interface EmployeeSummary {
  name: string;
  tags?: string[];
}

export interface DailyAttendanceRecord {
  timestamp: string;
  employee_name: string;
  confidence_distance?: string;
  emotion?: string;
  liveness_status?: string;
}

export interface AttendanceSummary {
  total_count: number;
  unique_employees: number;
  avg_confidence: number;
  liveness_real_count: number;
  liveness_spoof_count: number;
}
