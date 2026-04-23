import { valueTicks, SvgAxes, M } from './TimeseriesChart'
import { OBS_COLOR } from './theme'

const W = 340
const H = 280
const IW = W - M.left - M.right
const IH = H - M.top - M.bottom

export default function ScatterPlot({ dx, dy, xLabel, yLabel }) {
  if (!dx?.length) return null
  const xMin = Math.min(...dx), xMax = Math.max(...dx)
  const yMin = Math.min(...dy), yMax = Math.max(...dy)
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1
  const sx = v => M.left + ((v - xMin) / xRange) * IW
  const sy = v => M.top + IH - ((v - yMin) / yRange) * IH
  const xTicks = valueTicks(xMin, xMax, 4)
  const yTicks = valueTicks(yMin, yMax, 4)
  const x0 = sx(Math.max(xMin, Math.min(xMax, 0)))
  const y0 = sy(Math.max(yMin, Math.min(yMax, 0)))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" style={{ maxWidth: W }}>
      <rect x={M.left} y={M.top} width={IW} height={IH} fill="#191e2b" />
      {yTicks.map(v => <line key={v} x1={M.left} x2={W - M.right} y1={sy(v)} y2={sy(v)} className="grid-line" />)}
      {xTicks.map(v => <line key={`x${v}`} x1={sx(v)} x2={sx(v)} y1={M.top} y2={M.top + IH} className="grid-line" />)}
      <line x1={x0} x2={x0} y1={M.top} y2={M.top + IH} className="grid-line" />
      <line x1={M.left} x2={W - M.right} y1={y0} y2={y0} className="grid-line" />
      {dx.map((dxi, i) => (
        <g key={i}>
          <circle cx={sx(dxi)} cy={sy(dy[i])} r="4" fill={OBS_COLOR} opacity="0.7" />
          <text x={sx(dxi) + 5} y={sy(dy[i]) - 4} fontSize="9" fill={OBS_COLOR} opacity="0.7">{i}</text>
        </g>
      ))}
      <SvgAxes
        xTicks={xTicks} yTicks={yTicks}
        x={sx} y={sy}
        m={M} vw={W} iw={IW} ih={IH}
        formatX={v => Math.round(v)} formatY={v => v.toFixed(1)}
        xLabel={xLabel} yLabel={yLabel}
      />
    </svg>
  )
}
