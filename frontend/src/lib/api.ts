// frontend/src/lib/api.ts
// Typed API client for the OceanEmbed FastAPI backend

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://oceanembed-1-555s.onrender.com";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// Health
export const getHealth = () => apiFetch<{ status: string }>("/api/health");

// Models
export const getModels = () =>
  apiFetch<import("@/types").ModelsResponse>("/api/models");

// Predict — dates
export const getTestDates = () =>
  apiFetch<{ dates: string[]; n_samples: number }>("/api/predict/dates");

// Predict — inference
export const runPredict = (body: import("@/types").PredictRequest) =>
  apiFetch<import("@/types").PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });

// Results
export const getEvaluation = () =>
  apiFetch<import("@/types").EvaluationSummary>("/api/results/evaluation");

export const getPerDepthMetrics = () =>
  apiFetch<import("@/types").PerDepthRow[]>("/api/results/per-depth");

export const getTrainingHistory = () =>
  apiFetch<Record<string, import("@/types").TrainingHistory>>(
    "/api/results/training-history"
  );

export const getArgoProfiles = () =>
  apiFetch<{ profiles: import("@/types").ArgoProfile[]; count: number }>(
    "/api/results/argo/profiles"
  );

export const getArgoObservations = (profileId?: string, limit = 500) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (profileId) params.set("profile_id", profileId);
  return apiFetch<{
    data: import("@/types").ArgoObservation[];
    count: number;
  }>(`/api/results/argo?${params}`);
};

export const getArgoAggregate = () =>
  apiFetch<import("@/types").ArgoAggregate>("/api/results/argo/aggregate");
