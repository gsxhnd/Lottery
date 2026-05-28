import { useMemo, useState } from "react"

import type { DrawRecord, WinningStatsResponse } from "@/types/lottery"

type DataRecordsPageProps = {
  recentLimit: number
  vizLoading: boolean
  vizError: string | null
  vizStats: WinningStatsResponse | null
  setRecentLimit: (value: number) => void
  loadVizData: (limit: number) => Promise<void>
}

function drawMatches(record: DrawRecord, opts: { issueKeyword: string; blueBall: string; redSumMin: string; redSumMax: string; redBall: string }) {
  const { issueKeyword, blueBall, redSumMin, redSumMax, redBall } = opts
  if (issueKeyword && !record.issue.includes(issueKeyword.trim())) return false
  if (blueBall) {
    const target = Number(blueBall)
    if (!Number.isNaN(target) && record.blue_ball !== target) return false
  }
  if (redBall) {
    const target = Number(redBall)
    if (!Number.isNaN(target) && !record.red_balls.includes(target)) return false
  }
  if (redSumMin) {
    const min = Number(redSumMin)
    if (!Number.isNaN(min) && record.red_sum < min) return false
  }
  if (redSumMax) {
    const max = Number(redSumMax)
    if (!Number.isNaN(max) && record.red_sum > max) return false
  }
  return true
}

export function DataRecordsPage(props: DataRecordsPageProps) {
  const { recentLimit, vizLoading, vizError, vizStats, setRecentLimit, loadVizData } = props
  const [issueKeyword, setIssueKeyword] = useState("")
  const [blueBall, setBlueBall] = useState("")
  const [redBall, setRedBall] = useState("")
  const [redSumMin, setRedSumMin] = useState("")
  const [redSumMax, setRedSumMax] = useState("")

  const filtered = useMemo(() => {
    if (!vizStats) return []
    return [...vizStats.recent_draws]
      .reverse()
      .filter((record) => drawMatches(record, { issueKeyword, blueBall, redSumMin, redSumMax, redBall }))
  }, [vizStats, issueKeyword, blueBall, redSumMin, redSumMax, redBall])

  return (
    <main>
      <section className="panel">
        <div className="viz-actions">
          <select value={recentLimit} onChange={(e) => setRecentLimit(Number(e.target.value))}>
            <option value={60}>最近 60 期</option>
            <option value={120}>最近 120 期</option>
            <option value={240}>最近 240 期</option>
            <option value={500}>最近 500 期</option>
          </select>
          <button type="button" className="btn btn--secondary" onClick={() => void loadVizData(recentLimit)} disabled={vizLoading}>
            {vizLoading ? "加载中..." : "刷新数据"}
          </button>
        </div>

        <div className="filter-grid">
          <input value={issueKeyword} onChange={(e) => setIssueKeyword(e.target.value)} placeholder="筛选期号，如 2026" />
          <input value={redBall} onChange={(e) => setRedBall(e.target.value)} placeholder="包含红球，如 12" />
          <input value={blueBall} onChange={(e) => setBlueBall(e.target.value)} placeholder="蓝球等于，如 08" />
          <input value={redSumMin} onChange={(e) => setRedSumMin(e.target.value)} placeholder="和值最小值" />
          <input value={redSumMax} onChange={(e) => setRedSumMax(e.target.value)} placeholder="和值最大值" />
        </div>

        {vizError ? <p className="status error">{vizError}</p> : null}
        {!vizError && vizLoading ? <p className="status">正在读取开奖记录...</p> : null}

        {vizStats ? (
          <>
            <p className="status">{`共 ${vizStats.recent_draws.length} 条，筛选后 ${filtered.length} 条`}</p>
            <div className="records-table-wrap">
              <table className="records-table">
                <thead>
                  <tr>
                    <th>期号</th>
                    <th>日期</th>
                    <th>红球</th>
                    <th>蓝球</th>
                    <th>和值</th>
                    <th>奇偶比</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((row) => {
                    const odd = row.red_balls.filter((n) => n % 2 === 1).length
                    const even = row.red_balls.length - odd
                    return (
                      <tr key={row.issue}>
                        <td>{row.issue}</td>
                        <td>{row.date}</td>
                        <td className="records-table__mono">{row.red_balls.map((n) => String(n).padStart(2, "0")).join(" ")}</td>
                        <td className="records-table__mono">{String(row.blue_ball).padStart(2, "0")}</td>
                        <td>{row.red_sum}</td>
                        <td>{`${odd}:${even}`}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </section>
    </main>
  )
}
