import * as d3 from "d3"
import { useEffect, useRef, useState } from "react"

import type { DrawRecord } from "@/types/lottery"

type TrendDatum = DrawRecord & { index: number; movingAvg: number }

export function RecentTrendChart(props: { draws: DrawRecord[] }) {
  const { draws } = props
  const ref = useRef<SVGSVGElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const [hovered, setHovered] = useState<TrendDatum | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (!ref.current || draws.length === 0) return
    const width = 760
    const height = 300
    const margin = { top: 28, right: 28, bottom: 44, left: 48 }
    const indexed: TrendDatum[] = draws.map((d, i, arr) => {
      const start = Math.max(0, i - 4)
      const sample = arr.slice(start, i + 1)
      const avg = d3.mean(sample, (r) => r.red_sum) ?? d.red_sum
      return { ...d, index: i + 1, movingAvg: avg }
    })

    const svg = d3.select(ref.current)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${width} ${height}`)

    const x = d3.scaleLinear().domain([1, indexed.length]).range([margin.left, width - margin.right])
    const yMin = d3.min(indexed, (d) => d.red_sum) ?? 0
    const yMax = d3.max(indexed, (d) => d.red_sum) ?? 0
    const y = d3
      .scaleLinear()
      .domain([Math.max(0, yMin - 8), yMax + 8])
      .nice()
      .range([height - margin.bottom, margin.top])

    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(Math.min(10, indexed.length)).tickFormat((v) => `#${v}`))
      .call((g) => g.selectAll("text").attr("fill", "#a8b0c4"))
      .call((g) => g.selectAll("path,line").attr("stroke", "#3b4456"))

    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(7))
      .call((g) => g.selectAll("text").attr("fill", "#a8b0c4"))
      .call((g) => g.selectAll("path,line").attr("stroke", "#3b4456"))

    const line = d3
      .line<TrendDatum>()
      .x((d) => x(d.index))
      .y((d) => y(d.red_sum))
      .curve(d3.curveMonotoneX)

    const avgLine = d3
      .line<TrendDatum>()
      .x((d) => x(d.index))
      .y((d) => y(d.movingAvg))
      .curve(d3.curveMonotoneX)

    svg.append("path").datum(indexed).attr("fill", "none").attr("stroke", "#8b5cf6").attr("stroke-width", 2.4).attr("d", line)
    svg
      .append("path")
      .datum(indexed)
      .attr("fill", "none")
      .attr("stroke", "#22d3ee")
      .attr("stroke-width", 1.8)
      .attr("stroke-dasharray", "5 5")
      .attr("d", avgLine)

    svg
      .append("g")
      .selectAll("circle")
      .data(indexed)
      .join("circle")
      .attr("cx", (d) => x(d.index))
      .attr("cy", (d) => y(d.red_sum))
      .attr("r", 3.5)
      .attr("fill", "#c4b5fd")
      .style("cursor", "pointer")
      .on("mouseenter", (event, d) => {
        const [mx, my] = d3.pointer(event, wrapRef.current)
        setHovered(d)
        setTooltipPos({ x: mx + 12, y: my + 12 })
      })
      .on("mousemove", (event) => {
        const [mx, my] = d3.pointer(event, wrapRef.current)
        setTooltipPos({ x: mx + 12, y: my + 12 })
      })
      .on("mouseleave", () => setHovered(null))
  }, [draws])

  return (
    <div ref={wrapRef} className="chart-wrap">
      <svg ref={ref} className="chart-svg" />
      {hovered ? (
        <div className="chart-tooltip" style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y}px` }}>
          <p>{`期号: ${hovered.issue}`}</p>
          <p>{`日期: ${hovered.date}`}</p>
          <p>{`红球和值: ${hovered.red_sum}`}</p>
          <p>{`5期均值: ${hovered.movingAvg.toFixed(1)}`}</p>
          <p>{`蓝球: ${String(hovered.blue_ball).padStart(2, "0")}`}</p>
        </div>
      ) : null}
    </div>
  )
}
