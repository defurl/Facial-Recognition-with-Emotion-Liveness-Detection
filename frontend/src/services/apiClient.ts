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
} from "../types/api";

const BASE_URL = process.env.REACT_APP_API_BASE_URL ?? "http://localhost:8000";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }
  return response.json() as Promise<T>;
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
  return handleResponse<RegisterResponse>(res);
}

export async function fetchThreshold(): Promise<ThresholdPayload> {
  const res = await fetch(`${BASE_URL}/threshold`);
  return handleResponse(res);
}

export async function updateThreshold(payload: ThresholdPayload): Promise<ThresholdPayload> {
  const res = await fetch(`${BASE_URL}/threshold`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function logAttendance(event: AttendanceEvent): Promise<void> {
  await fetch(`${BASE_URL}/attendance/log`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });
}

export async function fetchEmployees(): Promise<EmployeeSummary[]> {
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
