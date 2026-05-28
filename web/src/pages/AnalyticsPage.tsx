import { FrequencyChart } from "@/components/charts/FrequencyChart"
import { OddEvenTrendChart } from "@/components/charts/OddEvenTrendChart"
import { RecentTrendChart } from "@/components/charts/RecentTrendChart"
import { SumDistributionChart } from "@/components/charts/SumDistributionChart"
import type { WinningStatsResponse } from "@/types/lottery"

type AnalyticsPageProps = {
  recentLimit: number
  vizLoading: boolean
  vizError: string | null
  vizStats: WinningStatsResponse | null
  setRecentLimit: (value: number) => void
  loadVizData: (limit: number) => Promise<void>
}

export function AnalyticsPage(props: AnalyticsPageProps) {
  const { recentLimit, vizLoading, vizError, vizStats, setRecentLimit, loadVizData } = props

  return (
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
              <div className="chart-legend">
                <span>
                  <i className="legend-dot legend-dot--line" />
                  红球和值
                </span>
                <span>
                  <i className="legend-dot legend-dot--avg" />
                  5期移动均值
                </span>
              </div>
              <RecentTrendChart draws={vizStats.recent_draws} />
            </section>
            <section className="chart-grid-2">
              <article className="card">
                <h3>红球和值分布（区间统计）</h3>
                <SumDistributionChart draws={vizStats.recent_draws} />
              </article>
              <article className="card">
                <h3>最近各期奇偶比例（红球）</h3>
                <OddEvenTrendChart draws={vizStats.recent_draws} />
              </article>
            </section>
          </>
        ) : null}
      </section>
    </main>
  )
}
