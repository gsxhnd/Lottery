export type HealthResponse = { status: string; records: number; model_loaded: boolean }

export type ModelInfo = { id: string; path: string; timestamp?: string | null }

export type PredictionResponse = {
  model_dir: string
  model_timestamp?: string | null
  input_window: { seq_len: number; issues: string[]; last_issue: string }
  prediction: { red_balls: number[]; blue_ball: number }
  normalized: unknown
  summary_path?: string | null
}

export type BallFrequency = { ball: number; count: number }

export type DrawRecord = {
  issue: string
  date: string
  red_balls: number[]
  blue_ball: number
  red_sum: number
}

export type WinningStatsResponse = {
  total_records: number
  issue_range: { start: string | null; end: string | null }
  red_frequencies: BallFrequency[]
  blue_frequencies: BallFrequency[]
  recent_draws: DrawRecord[]
}
