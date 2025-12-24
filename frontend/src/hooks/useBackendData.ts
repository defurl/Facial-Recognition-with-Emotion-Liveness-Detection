import { useEffect, useState, useCallback } from "react";
import { fetchEmployees, fetchThreshold, pingHealth } from "../services/apiClient";

export type BackendHealthState = "initializing" | "ok" | "unavailable";

export function useBackendData() {
  const [health, setHealth] = useState<BackendHealthState>("initializing");
  const [threshold, setThreshold] = useState<number | null>(null);
  const [employees, setEmployees] = useState<string[]>([]);

  const refresh = useCallback(() => {
    pingHealth()
      .then(() => setHealth("ok"))
      .catch(() => setHealth("unavailable"));

    fetchThreshold()
      .then((payload) => setThreshold(payload.threshold))
      .catch(() => setThreshold(null));

    fetchEmployees()
      .then((data) => setEmployees(data.employees))
      .catch(() => setEmployees([]));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    health,
    threshold,
    employees,
    refresh,
  };
}
