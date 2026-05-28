import { useEffect, useMemo, useState } from "react"

type HealthResponse = { status: string; records: number; model_loaded: boolean }
type ModelInfo = { id: string; path: string; timestamp?: string | null }
type PredictionResponse = {
  model_dir: string
  model_timestamp?: string | null
  input_window: { seq_len: number; issues: string[]; last_issue: string }
  prediction: { red_balls: number[]; blue_ball: number }
  normalized: unknown
  summary_path?: string | null
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options,
  })
  const text = await res.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { detail: text }
    }
  }
  if (!res.ok) {
    const detail = typeof body === "object" && body !== null && "detail" in body ? String((body as { detail: unknown }).detail) : res.statusText
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return body as T
}

function Ball(props: { num?: number; type: "red" | "blue"; size?: "sm" | "md" | "lg"; delay?: number }) {
  const { num, type, size = "md", delay = 0 } = props
  return (
    <span className={`ball ball--${type} ball--${size}`} style={{ animationDelay: `${delay}ms` }}>
      {num ? String(num).padStart(2, "0") : ""}
    </span>
  )
}

export default function App() {
  const [healthStatus, setHealthStatus] = useState<"loading" | "ok" | "error">("loading")
  const [healthText, setHealthText] = useState("连接中…")
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModelPath, setSelectedModelPath] = useState<string | null>(null)
  const [loadedModelPath, setLoadedModelPath] = useState<string | null>(null)
  const [saveSummary, setSaveSummary] = useState(false)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [toast, setToast] = useState<{ message: string; isError: boolean } | null>(null)
  const modelItems = Array.isArray(models) ? models : []

  const canPredict = useMemo(() => Boolean(loadedModelPath || selectedModelPath) && !isPredicting, [isPredicting, loadedModelPath, selectedModelPath])

  useEffect(() => {
    let timer: number | undefined
    if (toast) timer = window.setTimeout(() => setToast(null), 4000)
    return () => {
      if (timer) window.clearTimeout(timer)
    }
  }, [toast])

  const showToast = (message: string, isError = false) => setToast({ message, isError })

  const refreshHealth = async () => {
    try {
      const h = await api<HealthResponse>("/health")
      const modelNote = h.model_loaded ? " · 模型已加载" : ""
      setHealthStatus("ok")
      setHealthText(`${h.records.toLocaleString()} 期数据${modelNote}`)
    } catch {
      setHealthStatus("error")
      setHealthText("API 不可用")
    }
  }

  const fetchCurrentModel = async () => {
    try {
      const current = await api<ModelInfo>("/models/current")
      setLoadedModelPath(current.path)
      setSelectedModelPath((v) => v ?? current.path)
    } catch {
      setLoadedModelPath(null)
    }
  }

  const loadModels = async () => {
    try {
      const raw = await api<unknown>("/models")
      const list = Array.isArray(raw) ? (raw as ModelInfo[]) : []
      setModels(list)
      setSelectedModelPath((current) => current ?? list[0]?.path ?? null)
    } catch (error) {
      setModels([])
      showToast(error instanceof Error ? error.message : "加载模型列表失败", true)
    }
  }

  const loadSelectedModel = async () => {
    if (!selectedModelPath) return
    setIsLoadingModel(true)
    try {
      const info = await api<ModelInfo>("/models/load", { method: "POST", body: JSON.stringify({ model: selectedModelPath }) })
      setLoadedModelPath(info.path)
      showToast("模型已加载")
      await refreshHealth()
    } catch (error) {
      showToast(error instanceof Error ? error.message : "加载模型失败", true)
    } finally {
      setIsLoadingModel(false)
    }
  }

  const runPredict = async () => {
    setIsPredicting(true)
    try {
      const body: { save_summary: boolean; model?: string } = { save_summary: saveSummary }
      if (!loadedModelPath && selectedModelPath) body.model = selectedModelPath
      const data = await api<PredictionResponse>("/predict", { method: "POST", body: JSON.stringify(body) })
      setPrediction(data)
      if (data.model_dir) setLoadedModelPath(data.model_dir)
      showToast("预测完成")
      await refreshHealth()
    } catch (error) {
      showToast(error instanceof Error ? error.message : "预测失败", true)
    } finally {
      setIsPredicting(false)
    }
  }

  const reloadData = async () => {
    try {
      const data = await api<{ records: number }>("/data/reload", { method: "POST" })
      showToast(`已刷新 ${data.records.toLocaleString()} 期数据`)
      await refreshHealth()
    } catch (error) {
      showToast(error instanceof Error ? error.message : "刷新失败", true)
    }
  }

  useEffect(() => {
    void (async () => {
      await refreshHealth()
      await Promise.all([loadModels(), fetchCurrentModel()])
    })()
  }, [])

  return (
    <>
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-glow bg-glow--red" aria-hidden="true" />
      <div className="bg-glow bg-glow--blue" aria-hidden="true" />
      <div className="app">
        <header className="header">
          <div className="brand">
            <div className="brand__icon" aria-hidden="true">
              <Ball type="red" size="sm" />
              <Ball type="blue" size="sm" />
            </div>
            <div>
              <h1 className="brand__title">Lottery</h1>
              <p className="brand__sub">LSTM 预测控制台</p>
            </div>
          </div>
          <div className="header__meta">
            <div className={`pill pill--${healthStatus}`}>
              <span className="pill__dot" />
              <span className="pill__text">{healthText}</span>
            </div>
            <a className="link-api" href="/docs" target="_blank" rel="noreferrer">
              API 文档
            </a>
          </div>
        </header>

        <main className="layout">
          <aside className="panel panel--models">
            <div className="panel__head">
              <h2>模型</h2>
              <button className="btn btn--ghost btn--icon" type="button" title="刷新列表" onClick={() => void loadModels()}>
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12a9 9 0 1 1-2.64-6.36" />
                  <path d="M21 3v6h-6" />
                </svg>
              </button>
            </div>
            <p className="panel__hint">选择训练产物并加载到内存</p>
            <div className="model-list" role="listbox" aria-label="可用模型">
              {modelItems.length === 0 ? (
                <p className="model-list--empty">
                  暂无模型，请先运行 <code>lottery train</code>
                </p>
              ) : (
                modelItems.map((m) => (
                  <button
                    key={m.path}
                    type="button"
                    className={`model-card ${m.path === selectedModelPath ? "model-card--selected" : ""}`}
                    onClick={() => setSelectedModelPath(m.path)}
                  >
                    <span className="model-card__id">{m.id}</span>
                    <span className="model-card__time">{m.timestamp ?? "—"}</span>
                  </button>
                ))
              )}
            </div>
            <div className="panel__actions">
              <button className="btn btn--secondary" type="button" disabled={!selectedModelPath || isLoadingModel} onClick={() => void loadSelectedModel()}>
                {isLoadingModel ? "加载中..." : "加载模型"}
              </button>
            </div>
            <div className={`loaded-model ${loadedModelPath ? "loaded-model--active" : "loaded-model--empty"}`}>
              <span className="loaded-model__label">当前</span>
              <span className="loaded-model__value">{loadedModelPath ?? "未加载"}</span>
            </div>
          </aside>

          <section className="panel panel--predict">
            <div className="panel__head">
              <h2>预测</h2>
              <label className="checkbox">
                <input type="checkbox" checked={saveSummary} onChange={(e) => setSaveSummary(e.target.checked)} />
                <span>保存到 summaries</span>
              </label>
            </div>

            {!prediction ? (
              <div className="result-empty">
                <div className="result-empty__balls" aria-hidden="true">
                  <Ball type="red" size="lg" />
                  <Ball type="red" size="lg" />
                  <Ball type="red" size="lg" />
                  <Ball type="blue" size="lg" />
                </div>
                <p>加载模型后点击「运行预测」</p>
              </div>
            ) : (
              <div className="result">
                <div className="balls-row">
                  <div className="balls-group">
                    <span className="balls-label">红球</span>
                    <div className="balls">
                      {prediction.prediction.red_balls.map((n, i) => (
                        <Ball key={`${n}-${i}`} num={n} type="red" size="md" delay={i * 60} />
                      ))}
                    </div>
                  </div>
                  <div className="balls-group balls-group--blue">
                    <span className="balls-label">蓝球</span>
                    <div className="balls">
                      <Ball num={prediction.prediction.blue_ball} type="blue" size="md" delay={360} />
                    </div>
                  </div>
                </div>
                <dl className="meta-grid">
                  <div>
                    <dt>模型</dt>
                    <dd>{prediction.model_dir}</dd>
                  </div>
                  <div>
                    <dt>训练时间</dt>
                    <dd>{prediction.model_timestamp ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>输入窗口</dt>
                    <dd>{`${prediction.input_window.seq_len} 期 · ${prediction.input_window.issues.join(", ")}`}</dd>
                  </div>
                  <div>
                    <dt>最近一期</dt>
                    <dd>{prediction.input_window.last_issue}</dd>
                  </div>
                </dl>
                <details className="details">
                  <summary>归一化输出</summary>
                  <pre className="code-block">{JSON.stringify(prediction.normalized, null, 2)}</pre>
                </details>
                {prediction.summary_path ? <p className="summary-path">{`已保存: ${prediction.summary_path}`}</p> : null}
              </div>
            )}

            <div className="panel__actions panel__actions--center">
              <button className={`btn btn--primary btn--lg ${isPredicting ? "is-loading" : ""}`} type="button" disabled={!canPredict} onClick={() => void runPredict()}>
                <span className="btn__label">运行预测</span>
                <span className={`btn__spinner ${isPredicting ? "" : "hidden"}`} aria-hidden="true" />
              </button>
              <button className="btn btn--ghost" type="button" onClick={() => void reloadData()}>
                刷新历史数据
              </button>
            </div>
          </section>
        </main>
      </div>

      {toast ? (
        <div className={`toast ${toast.isError ? "toast--error" : ""}`} role="status" aria-live="polite">
          {toast.message}
        </div>
      ) : null}
    </>
  )
}
