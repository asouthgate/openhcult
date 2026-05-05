import { useState } from 'react'
import { OBS_COLOR, PENDING_COLOR, PALETTE } from './theme'

export const M = { top: 20, right: 24, bottom: 52, left: 60 }
export const VW = 900
export const VH = 380
export const IW = VW - M.left - M.right
export const IH = VH - M.top - M.bottom

const pts = (arr, fx, fy) => arr.map(p => `${fx(p).toFixed(1)},${fy(p).toFixed(1)}`).join(' ')

export function timeTicks(tMin, tMax, n) {
  const step = (tMax - tMin) / n
  return Array.from({ length: n + 1 }, (_, i) => tMin + i * step)
}

export function valueTicks(vMin, vMax, n) {
  const range = vMax - vMin || 1
  const rawStep = range / n
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const step = Math.ceil(rawStep / mag) * mag
  const first = Math.floor(vMin / step) * step
  const ticks = []
  for (let v = first; v <= vMax + step * 0.5; v += step) ticks.push(v)
  return ticks.filter(v => v >= vMin - step * 0.1 && v <= vMax + step * 1.1)
}

function fmtTime(ms, rangeMs) {
  const d = new Date(ms)
  if (rangeMs <= 24 * 3600 * 1000) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export function SvgAxes({ xTicks, yTicks, x, y, m, vw, iw, ih, formatX, formatY, xLabel, yLabel }) {
  return (
    <>
      <line x1={m.left} x2={vw - m.right} y1={m.top + ih} y2={m.top + ih} className="axis-line" />
      <line x1={m.left} x2={m.left} y1={m.top} y2={m.top + ih} className="axis-line" />
      {xTicks.map(v => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={m.top + ih} y2={m.top + ih + 5} className="axis-line" />
          <text x={x(v)} y={m.top + ih + 18} className="axis-label" textAnchor="middle">{formatX(v)}</text>
        </g>
      ))}
      {yTicks.map(v => (
        <g key={v}>
          <line x1={m.left - 5} x2={m.left} y1={y(v)} y2={y(v)} className="axis-line" />
          <text x={m.left - 9} y={y(v) + 4} className="axis-label" textAnchor="end">{formatY(v)}</text>
        </g>
      ))}
      {xLabel && (
        <text x={m.left + iw / 2} y={m.top + ih + m.bottom - 6} className="axis-label" textAnchor="middle">{xLabel}</text>
      )}
      {yLabel && (
        <text x={14} y={m.top + ih / 2} className="axis-label" textAnchor="middle"
          transform={`rotate(-90, 14, ${m.top + ih / 2})`}>{yLabel}</text>
      )}
    </>
  )
}

export function TimeseriesChart({ series, bands = [], observations, rangeMs, onTimePick, pendingTime, yLabel, eventWindowOffset, eventWindowWidth, hideObsLegend }) {
  const [cursor, setCursor] = useState(null)

  const allT = series.flatMap(s => s.points.map(p => p.t))
  const allV = [
    ...series.flatMap(s => s.points.map(p => p.v)),
    ...bands.flatMap(b => b.points.flatMap(p => [p.lo, p.hi])),
  ]
  if (!allT.length) return null

  const tMin = Math.min(...allT)
  const tMax = Math.max(...allT)
  const vMin = Math.min(...allV)
  const vMax = Math.max(...allV)
  const vRange = vMax - vMin || 1

  const x = t => M.left + ((t - tMin) / (tMax - tMin || 1)) * IW
  const y = v => M.top + IH - ((v - vMin) / vRange) * IH

  const xTicks = timeTicks(tMin, tMax, 6)
  const yTicks = valueTicks(vMin, vMax, 8)
  const yStep = yTicks.length > 1 ? Math.abs(yTicks[1] - yTicks[0]) : 1
  const yDecimals = yStep >= 1 ? 0 : Math.max(0, -Math.floor(Math.log10(yStep)) + 1)

  const inRange = t => t >= tMin && t <= tMax

  const svgCoordToTime = (clientX, rect) => {
    const svgX = ((clientX - rect.left) / rect.width) * VW
    if (svgX < M.left || svgX > VW - M.right) return null
    return { x: svgX, t: tMin + ((svgX - M.left) / IW) * (tMax - tMin) }
  }

  const handleMouseMove = e => {
    setCursor(svgCoordToTime(e.clientX, e.currentTarget.getBoundingClientRect()))
  }

  const handleClick = e => {
    const coord = cursor ?? svgCoordToTime(e.clientX, e.currentTarget.getBoundingClientRect())
    if (coord) onTimePick?.(coord.t)
  }

  return (
    <div className="chart-wrap">
      <svg
        viewBox={`0 0 ${VW} ${VH}`}
        className="chart-svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setCursor(null)}
        onClick={handleClick}
      >
        <rect x={M.left} y={M.top} width={IW} height={IH} fill="#191e2b" />
        {yTicks.map(v => (
          <line key={v} x1={M.left} x2={VW - M.right} y1={y(v)} y2={y(v)} className="grid-line" />
        ))}
        {xTicks.map(v => (
          <line key={`x${v}`} x1={x(v)} x2={x(v)} y1={M.top} y2={M.top + IH} className="grid-line" />
        ))}

        {bands.map((b, i) => {
          const upper = pts(b.points, p => x(p.t), p => y(p.hi))
          const lower = pts([...b.points].reverse(), p => x(p.t), p => y(p.lo))
          return (
            <g key={i}>
              <polygon points={`${upper} ${lower}`} fill={b.color} opacity="0.25" />
              <polyline points={upper} fill="none" stroke={b.color} strokeWidth="1.5" strokeDasharray="3 3" opacity="0.7" />
              <polyline points={lower} fill="none" stroke={b.color} strokeWidth="1.5" strokeDasharray="3 3" opacity="0.7" />
            </g>
          )
        })}

        {series.map(s => (
          <polyline
            key={s.label}
            points={pts(s.points, p => x(p.t), p => y(p.v))}
            fill="none"
            stroke={s.color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            opacity="0.9"
          />
        ))}

        {(observations ?? []).filter(o => inRange(+new Date(o.observed_at))).map(o => {
          const confirmed = o.note?.includes('WATER') && !o.note?.includes('AUTO')
          const t0 = +new Date(o.observed_at)
          const showWindows = eventWindowOffset != null && eventWindowWidth != null
          const bStart = t0 - eventWindowOffset - eventWindowWidth
          const bEnd = t0 - eventWindowOffset
          const aStart = t0 + eventWindowOffset
          const aEnd = t0 + eventWindowOffset + eventWindowWidth
          const clamp = t => Math.max(tMin, Math.min(tMax, t))
          return (
            <g key={o.id}>
              {showWindows && (
                <>
                  <rect
                    x={x(clamp(bStart))} y={M.top}
                    width={Math.max(0, x(clamp(bEnd)) - x(clamp(bStart)))}
                    height={IH}
                    fill={OBS_COLOR} opacity="0.12"
                  />
                  <rect
                    x={x(clamp(aStart))} y={M.top}
                    width={Math.max(0, x(clamp(aEnd)) - x(clamp(aStart)))}
                    height={IH}
                    fill={OBS_COLOR} opacity="0.12"
                  />
                </>
              )}
              <line
                x1={x(t0)} x2={x(t0)}
                y1={M.top} y2={M.top + IH}
                stroke={OBS_COLOR} strokeWidth="1.5" strokeDasharray="4 3" opacity="0.8"
              />
              {confirmed && (
                <text x={x(t0)} y={M.top - 4} textAnchor="middle"
                  fontSize="13" fill={OBS_COLOR} opacity="0.9">★</text>
              )}
            </g>
          )
        })}

        <SvgAxes
          xTicks={xTicks} yTicks={yTicks}
          x={x} y={y}
          m={M} vw={VW} iw={IW} ih={IH}
          formatX={t => fmtTime(t, rangeMs)} formatY={v => v.toFixed(yDecimals)}
          yLabel={yLabel}
        />

        {cursor && (
          <>
            <line x1={cursor.x} x2={cursor.x} y1={M.top} y2={M.top + IH} className="cursor-line" />
            <text
              x={cursor.x > VW / 2 ? cursor.x - 5 : cursor.x + 5}
              y={M.top + 13}
              textAnchor={cursor.x > VW / 2 ? 'end' : 'start'}
              className="cursor-label"
            >
              {fmtTime(cursor.t, rangeMs)}
            </text>
          </>
        )}

        {pendingTime != null && inRange(pendingTime) && (
          <>
            <line
              x1={x(pendingTime)} x2={x(pendingTime)}
              y1={M.top} y2={M.top + IH}
              stroke={PENDING_COLOR} strokeWidth="2" strokeDasharray="4 3"
            />
            <text
              x={x(pendingTime) > VW / 2 ? x(pendingTime) - 5 : x(pendingTime) + 5}
              y={M.top + 30}
              textAnchor={x(pendingTime) > VW / 2 ? 'end' : 'start'}
              fill={PENDING_COLOR} fontSize="11" fontFamily="monospace"
            >
              watering?
            </text>
          </>
        )}

<rect x={M.left} y={M.top} width={IW} height={IH} fill="transparent" />
      </svg>

      <div className="legend">
        {series.map(s => (
          <span key={s.label} className="legend-item">
            <span className="legend-dot" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        {!hideObsLegend && (
          <span className="legend-item">
            <span className="legend-dot" style={{ background: OBS_COLOR }} />
            recorded watering
          </span>
        )}
        {!hideObsLegend && (
          <span className="legend-item">
            <span className="legend-dot" style={{ background: PENDING_COLOR }} />
            candidate
          </span>
        )}
      </div>
    </div>
  )
}
