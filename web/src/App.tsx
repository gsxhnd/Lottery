import * as d3 from "d3"
import { useEffect, useMemo, useRef, useState } from "react"

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
type BallFrequency = { ball: number; count: number }
type DrawRecord = { issue: string; date: string; red_balls: number[]; blue_ball: number; red_sum: number }
type WinningStatsResponse = {
  total_records: number
  issue_range: { start: string | null; end: string | null }
  red_frequencies: BallFrequency[]
  blue_frequencies: BallFrequency[]
  recent_draws: DrawRecord[]
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

function FrequencyChart(props: { data: BallFrequency[]; color: string; title: string }) {
  const { data, color, title } = props
  const ref = useRef<SVGSVGElement | null>(null)

  useEffect(() => {
    if (!ref.current || data.length === 0) return
    const width = 720
    const height = 260
    const margin = { top: 24, right: 20, bottom: 40, left: 40 }
    const svg = d3.select(ref.current)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${width} ${height}`)

    const x = d3
      .scaleBand<number>()
      .domain(data.map((d) => d.ball))
      .range([margin.left, width - margin.right])
      .padding(0.15)
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d: BallFrequency) => d.count) ?? 0])
      .nice()
      .range([height - margin.bottom, margin.top])

    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickValues(data.filter((d) => d.ball % 2 === 1).map((d) => d.ball)).tickSizeOuter(0))
    svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0))

    svg
      .append("g")
      .selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", (d: BallFrequency) => x(d.ball) ?? 0)
      .attr("y", (d: BallFrequency) => y(d.count))
      .attr("width", x.bandwidth())
      .attr("height", (d: BallFrequency) => y(0) - y(d.count))
      .attr("rx", 5)
      .attr("fill", color)

    svg.append("text").attr("x", margin.left).attr("y", 16).attr("fill", "#f4f6fb").attr("font-size", 13).attr("font-weight", 600).text(title)
  }, [data, color, title])

  return <svg ref={ref} className="chart-svg" />
}

function RecentTrendChart(props: { draws: DrawRecord[] }) {
  const { draws } = props
  const ref = useRef<SVGSVGElement | null>(null)

  useEffect(() => {
    if (!ref.current || draws.length === 0) return
    const width = 720
    const height = 280
    const margin = { top: 24, right: 20, bottom: 40, left: 44 }
    const indexed = draws.map((d, i) => ({ ...d, index: i + 1 }))
    const svg = d3.select(ref.current)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${width} ${height}`)

    const x = d3.scaleLinear().domain([1, indexed.length]).range([margin.left, width - margin.right])
    const y = d3
      .scaleLinear()
      .domain(d3.extent(indexed, (d: (typeof indexed)[number]) => d.red_sum) as [number, number])
      .nice()
      .range([height - margin.bottom, margin.top])

    svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(8))
    svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6))

    const line = d3
      .line<(typeof indexed)[number]>()
      .x((d) => x(d.index))
      .y((d) => y(d.red_sum))
      .curve(d3.curveMonotoneX)

    svg.append("path").datum(indexed).attr("fill", "none").attr("stroke", "#8b5cf6").attr("stroke-width", 2.5).attr("d", line)
  }, [draws])

  return <svg ref={ref} className="chart-svg" />
}

export default function App() {
  const [page, setPage] = useState<"predict" | "viz">("predict")
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
  const [vizStats, setVizStats] = useState<WinningStatsResponse | null>(null)
  const [vizLoading, setVizLoading] = useState(false)
  const [vizError, setVizError] = useState<string | null>(null)
  const [recentLimit, setRecentLimit] = useState(120)
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

  const loadVizData = async (limit: number) => {
    setVizLoading(true)
    setVizError(null)
    try {
      const data = await api<WinningStatsResponse>(`/data/winning-stats?recent_limit=${limit}`)
      setVizStats(data)
    } catch (error) {
      setVizError(error instanceof Error ? error.message : "加载可视化数据失败")
    } finally {
      setVizLoading(false)
    }
  }

  useEffect(() => {
    void (async () => {
      await refreshHealth()
      await Promise.all([loadModels(), fetchCurrentModel()])
    })()
  }, [])

  useEffect(() => {
    if (page === "viz" && !vizStats && !vizLoading) {
      void loadVizData(recentLimit)
    }
  }, [page, vizLoading, vizStats, recentLimit])

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
            <button className="btn btn--ghost" type="button" onClick={() => setPage(page === "predict" ? "viz" : "predict")}>
              {page === "predict" ? "数据可视化" : "返回预测"}
            </button>
            <a className="link-api" href="/docs" target="_blank" rel="noreferrer">
              API 文档
            </a>
          </div>
        </header>

        {page === "predict" ? <main className="layout">
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
        </main> : (
          <main>
            <section className="panel">
              <div className="viz-actions">
                <select value={recentLimit} onChange={(e) => setRecentLimit(Number(e.target.value))}>
                  <option value={60}>最近 60 期</option>
                  <option value={120}>最近 120 期</option>
                  <option value={240}>最近 240 期</option>
                </select>
                <button type="button" className="btn btn--secondary" onClick={() => void loadVizData(recentLimit)} disabled={vizLoading}>
                  {vizLoading ? "加载中..." : "刷新可视化"}
                </button>
              </div>
              {vizError ? <p className="status error">{vizError}</p> : null}
              {!vizError && vizLoading ? <p className="status">正在读取 DuckDB 数据...</p> : null}
              {vizStats ? (
                <>
                  <section className="summary-grid">
                    <article className="card">
                      <h3>总记录数</h3>
                      <p>{vizStats.total_records.toLocaleString()}</p>
                    </article>
                    <article className="card">
                      <h3>起始期号</h3>
                      <p>{vizStats.issue_range.start ?? "-"}</p>
                    </article>
                    <article className="card">
                      <h3>最新期号</h3>
                      <p>{vizStats.issue_range.end ?? "-"}</p>
                    </article>
                  </section>
                  <section className="card chart-card">
                    <FrequencyChart data={vizStats.red_frequencies} color="#ef4444" title="红球号码出现频次（01-33）" />
                  </section>
                  <section className="card chart-card">
                    <FrequencyChart data={vizStats.blue_frequencies} color="#3b82f6" title="蓝球号码出现频次（01-16）" />
                  </section>
                  <section className="card chart-card">
                    <RecentTrendChart draws={vizStats.recent_draws} />
                  </section>
                </>
              ) : null}
            </section>
          </main>
        )}
      </div>

      {toast ? (
        <div className={`toast ${toast.isError ? "toast--error" : ""}`} role="status" aria-live="polite">
          {toast.message}
        </div>
      ) : null}
    </>
  )
}
