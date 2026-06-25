import { useMemo } from "react"

import type { DrawRecord } from "@/types/lottery"

type OddEvenRow = {
  issue: string
  odd: number
  even: number
}

export function OddEvenTrendChart(props: { draws: DrawRecord[] }) {
  const { draws } = props
  const rows = useMemo<OddEvenRow[]>(
    () =>
      draws.map((draw) => {
        const odd = draw.red_balls.filter((n) => n % 2 === 1).length
        return { issue: draw.issue, odd, even: draw.red_balls.length - odd }
      }),
    [draws],
  )

  return (
    <div className="odd-even">
      {rows.map((row) => {
        const oddPct = (row.odd / 6) * 100
        const evenPct = 100 - oddPct
        return (
          <div key={row.issue} className="odd-even__row">
            <div className="odd-even__issue">{row.issue}</div>
            <div className="odd-even__bar">
              <div className="odd-even__odd" style={{ width: `${oddPct}%` }} title={`奇数 ${row.odd}`} />
              <div className="odd-even__even" style={{ width: `${evenPct}%` }} title={`偶数 ${row.even}`} />
            </div>
            <div className="odd-even__meta">{`${row.odd}:${row.even}`}</div>
          </div>
        )
      })}
    </div>
  )
}
