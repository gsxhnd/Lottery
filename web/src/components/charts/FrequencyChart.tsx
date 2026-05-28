import * as d3 from "d3"
import { useEffect, useMemo, useRef, useState } from "react"

import type { BallFrequency } from "@/types/lottery"

export function FrequencyChart(props: { data: BallFrequency[]; color: string; title: string }) {
  const { data, color, title } = props
  const ref = useRef<SVGSVGElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const [hovered, setHovered] = useState<BallFrequency | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const total = useMemo(() => d3.sum(data, (d) => d.count), [data])

  useEffect(() => {
    if (!ref.current || data.length === 0) return
    const width = 760
    const height = 290
    const margin = { top: 40, right: 24, bottom: 44, left: 46 }
    const svg = d3.select(ref.current)
    svg.selectAll("*").remove()
    svg.attr("viewBox", `0 0 ${width} ${height}`)

    const x = d3
      .scaleBand<number>()
      .domain(data.map((d) => d.ball))
      .range([margin.left, width - margin.right])
      .padding(0.18)
    const yMax = d3.max(data, (d) => d.count) ?? 0
    const y = d3
      .scaleLinear()
      .domain([0, yMax * 1.1])
      .nice()
      .range([height - margin.bottom, margin.top])

    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickValues(data.filter((d) => d.ball % 2 === 1).map((d) => d.ball)).tickSizeOuter(0))
      .call((g) => g.selectAll("text").attr("fill", "#a8b0c4"))
      .call((g) => g.selectAll("path,line").attr("stroke", "#3b4456"))

    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(6).tickSizeOuter(0))
      .call((g) => g.selectAll("text").attr("fill", "#a8b0c4"))
      .call((g) => g.selectAll("path,line").attr("stroke", "#3b4456"))

    svg
      .append("g")
      .attr("stroke", "#263145")
      .attr("stroke-opacity", 0.5)
      .selectAll("line")
      .data(y.ticks(6))
      .join("line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d))

    const bars = svg
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
      .attr("opacity", 0.85)
      .style("cursor", "pointer")
      .on("mouseenter", (event, d) => {
        setHovered(d)
        const [mx, my] = d3.pointer(event, wrapRef.current)
        setTooltipPos({ x: mx + 12, y: my + 12 })
      })
      .on("mousemove", (event) => {
        const [mx, my] = d3.pointer(event, wrapRef.current)
        setTooltipPos({ x: mx + 12, y: my + 12 })
      })
      .on("mouseleave", () => setHovered(null))

    bars
      .transition()
      .duration(380)
      .attr("y", (d: BallFrequency) => y(d.count))
      .attr("height", (d: BallFrequency) => y(0) - y(d.count))

    svg.append("text").attr("x", margin.left).attr("y", 22).attr("fill", "#f4f6fb").attr("font-size", 13).attr("font-weight", 600).text(title)
  }, [data, color, title])

  return (
    <div ref={wrapRef} className="chart-wrap">
      <svg ref={ref} className="chart-svg" />
      {hovered ? (
        <div className="chart-tooltip" style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y}px` }}>
          <p>{`号码 ${String(hovered.ball).padStart(2, "0")}`}</p>
          <p>{`出现次数: ${hovered.count}`}</p>
          <p>{`占比: ${((hovered.count / Math.max(total, 1)) * 100).toFixed(2)}%`}</p>
        </div>
      ) : null}
    </div>
  )
}
