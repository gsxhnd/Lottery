import { Ball } from "@/components/lottery/Ball"
import type { ModelInfo, PredictionResponse } from "@/types/lottery"

type PredictPageProps = {
  modelItems: ModelInfo[]
  selectedModelPath: string | null
  loadedModelPath: string | null
  saveSummary: boolean
  isLoadingModel: boolean
  isPredicting: boolean
  prediction: PredictionResponse | null
  canPredict: boolean
  setSelectedModelPath: (value: string) => void
  setSaveSummary: (value: boolean) => void
  loadModels: () => Promise<void>
  loadSelectedModel: () => Promise<void>
  runPredict: () => Promise<void>
  reloadData: () => Promise<void>
}

export function PredictPage(props: PredictPageProps) {
  const {
    modelItems,
    selectedModelPath,
    loadedModelPath,
    saveSummary,
    isLoadingModel,
    isPredicting,
    prediction,
    canPredict,
    setSelectedModelPath,
    setSaveSummary,
    loadModels,
    loadSelectedModel,
    runPredict,
    reloadData,
  } = props

  return (
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
              <button key={m.path} type="button" className={`model-card ${m.path === selectedModelPath ? "model-card--selected" : ""}`} onClick={() => setSelectedModelPath(m.path)}>
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
            <div className="candidates">
              {(prediction.candidates ?? [prediction.prediction]).map((candidate, index) => (
                <div key={index} className={`candidate-row ${index === 0 ? "candidate-row--primary" : ""}`}>
                  <div className="candidate-row__head">
                    <span className="candidate-row__label">{index === 0 ? "主推" : `备选 ${index}`}</span>
                    <span className="candidate-row__hit-rate">{`命中率 ${candidate.hit_rate.toFixed(1)}%`}</span>
                  </div>
                  <p className="candidate-row__hit-detail">
                    {`红球均中 ${candidate.red_hit_avg.toFixed(2)} 个 · 蓝球 ${candidate.blue_hit_rate.toFixed(1)}%`}
                  </p>
                  <div className="balls-row">
                    <div className="balls-group">
                      <span className="balls-label">红球</span>
                      <div className="balls">
                        {candidate.red_balls.map((n, i) => (
                          <Ball key={`${index}-red-${n}-${i}`} num={n} type="red" size="md" delay={i * 60} />
                        ))}
                      </div>
                    </div>
                    <div className="balls-group balls-group--blue">
                      <span className="balls-label">蓝球</span>
                      <div className="balls">
                        <Ball num={candidate.blue_ball} type="blue" size="md" delay={360} />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
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
              <div>
                <dt>回测期数</dt>
                <dd>{prediction.backtest_periods > 0 ? `${prediction.backtest_periods} 期` : "—"}</dd>
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
  )
}
