import * as d3 from "d3"
import { useMemo } from "react"

import type { DrawRecord } from "@/types/lottery"

export function SumDistributionChart(props: { draws: DrawRecord[] }) {
  const { draws } = props
  const bins = useMemo(() => {
    const values = draws.map((d) => d.red_sum)
    if (values.length === 0) return []
    const min = d3.min(values) ?? 0
    const max = d3.max(values) ?? 0
    const thresholds = d3.range(Math.floor(min / 10) * 10, Math.ceil(max / 10) * 10 + 10, 10)
    return d3.bin().domain([min, max]).thresholds(thresholds)(values)
  }, [draws])

  const maxCount = d3.max(bins, (b) => b.length) ?? 0

  return (
    <div className="distribution">
      {bins.map((bin) => {
        const ratio = maxCount > 0 ? (bin.length / maxCount) * 100 : 0
        return (
          <div key={`${bin.x0}-${bin.x1}`} className="distribution-row">
            <div className="distribution-row__label">{`${Math.round(bin.x0 ?? 0)}-${Math.round(bin.x1 ?? 0)}`}</div>
            <div className="distribution-row__bar-wrap">
              <div className="distribution-row__bar" style={{ width: `${Math.max(ratio, 2)}%` }} />
            </div>
            <div className="distribution-row__count">{bin.length}</div>
          </div>
        )
      })}
    </div>
  )
}
