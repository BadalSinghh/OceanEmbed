// frontend/src/types/index.ts
// Shared TypeScript types matching backend API schemas

export interface ModelInfo {
  id: string;
  name: string;
  parameters: number;
  architecture: string;
  description: string;
  test_rmse: number;
  test_mae: number;
  test_r2: number;
  argo_rmse: number;
  argo_mae: number;
  pearson_r: number;
}

export interface Domain {
  region: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  resolution_deg: number;
  grid_shape: [number, number];
  depth_min_m: number;
  depth_max_m: number;
  depth_levels: number;
}

export interface InputChannel {
  id: string;
  name: string;
  unit: string;
  source: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
  target_depths: number[];
  domain: Domain;
  input_channels: InputChannel[];
}

export interface PredictRequest {
  model_id: string;
  sample_index: number;
  depth_index?: number | null;
}

export interface PredictResponse {
  model_id: string;
  model_name: string;
  date: string;
  sample_index: number;
  lats: number[];
  lons: number[];
  depths: number[];
  prediction: (number | null)[][] | (number | null)[][][];
  ground_truth: (number | null)[][] | (number | null)[][][];
  error: (number | null)[][] | (number | null)[][][];
  surface_inputs: (number | null)[][][];
}

export interface PerDepthRow {
  depth_m: number;
  cnn_rmse: number;
  cnn_mae: number;
  oceanembed_rmse: number;
  oceanembed_mae: number;
  oceanembed_r2: number;
  cbam_rmse: number;
  cbam_mae: number;
  cbam_r2: number;
}

export interface TrainingHistory {
  train_loss: number[];
  val_loss: number[];
  epoch_time_sec?: number[];
  best_val_loss?: number;
  total_training_time_sec?: number;
}

export interface ArgoProfile {
  profile_id: string;
  date: string;
  lat: number | null;
  lon: number | null;
}

export interface ArgoObservation {
  profile_id: string;
  date: string;
  lat: number | null;
  lon: number | null;
  depth_m: number | null;
  obs_temp: number | null;
  glorys_temp: number | null;
  cbam_temp: number | null;
  oe_temp: number | null;
  cnn_temp: number | null;
}

export interface ArgoAggregate {
  total_profiles: number;
  total_observations: number;
  test_period: { start: string; end: string };
  dataset: string;
  metrics: Record<string, { rmse: number; mae: number; pearson_r: number }>;
}

export interface EvaluationSummary {
  evaluation_timestamp_utc: string;
  models: Record<string, {
    parameters: number;
    training_time_sec: number;
    best_val_loss: number;
    test_rmse: number;
    test_mae: number;
    test_r2: number;
  }>;
  per_depth_test: Record<string, {
    cnn_rmse: number;
    cnn_mae: number;
    oceanembed_rmse: number;
    oceanembed_mae: number;
    oceanembed_r2: number;
    cbam_rmse: number;
    cbam_mae: number;
    cbam_r2: number;
  }>;
}
