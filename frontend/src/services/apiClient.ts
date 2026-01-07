import {
  VerifyRequest,
  VerifyResponse,
  RegisterRequest,
  RegisterResponse,
  ThresholdPayload,
  AttendanceEvent,
  EmployeeSummary,
  DailyAttendanceRecord,
  AttendanceSummary,
  DeleteEmployeeResponse,
  PoseValidateResponse,
} from "../types/api";

const BASE_URL = process.env.REACT_APP_API_BASE_URL ?? "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const payload = JSON.parse(text);
      if (payload.detail) {
        if (Array.isArray(payload.detail)) {
          message = payload.detail
            .map((entry: any) => (entry.msg ? `${entry.loc?.join(".")}: ${entry.msg}` : JSON.stringify(entry)))
            .join("; ");
        } else if (typeof payload.detail === "string") {
          message = payload.detail;
        }
      } else if (payload.message) {
        message = payload.message;
      }
    } catch (error) {
      // response was not JSON; keep raw text
    }
    throw new Error(message || `API error ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function pingHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse(res);
}

export async function verifyFace(payload: VerifyRequest): Promise<VerifyResponse> {
  const res = await fetch(`${BASE_URL}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function registerEmployee(payload: RegisterRequest): Promise<RegisterResponse> {
  const res = await fetch(`${BASE_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  console.log("Registration response status:", res.status);
  return handleResponse<RegisterResponse>(res);
}

export async function fetchThreshold(): Promise<ThresholdPayload> {
  const res = await fetch(`${BASE_URL}/threshold`);
  return handleResponse(res);
}

export async function updateThreshold(payload: ThresholdPayload): Promise<ThresholdPayload> {
  const res = await fetch(`${BASE_URL}/threshold`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function logAttendance(event: AttendanceEvent): Promise<void> {
  await fetch(`${BASE_URL}/attendance/mark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });
}

export async function fetchEmployees(): Promise<EmployeeSummary> {
  const res = await fetch(`${BASE_URL}/employees`);
  return handleResponse(res);
}

export async function fetchAttendanceToday(): Promise<DailyAttendanceRecord[]> {
  const res = await fetch(`${BASE_URL}/attendance/today`);
  return handleResponse(res);
}

export async function fetchAttendanceSummary(): Promise<AttendanceSummary> {
  const res = await fetch(`${BASE_URL}/attendance/summary`);
  return handleResponse(res);
}

export async function deleteEmployee(name: string): Promise<DeleteEmployeeResponse> {
  const res = await fetch(`${BASE_URL}/employees/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  return handleResponse(res);
}

export async function validatePose(
  image_b64: string,
  target_pose: "center" | "left" | "right"
): Promise<PoseValidateResponse> {
  const res = await fetch(`${BASE_URL}/pose/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_b64, target_pose }),
  });
  return handleResponse(res);
}

export interface ResetLivenessResponse {
  success: boolean;
  message: string;
  cleared_count: number;
}

export async function resetLivenessCache(employeeName?: string): Promise<ResetLivenessResponse> {
  const params = employeeName ? `?employee_name=${encodeURIComponent(employeeName)}` : "";
  const res = await fetch(`${BASE_URL}/liveness/reset${params}`, {
    method: "POST",
  });
  return handleResponse(res);
}
