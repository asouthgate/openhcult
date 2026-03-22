import { useState } from 'react'

const M = { top: 20, right: 24, bottom: 52, left: 60 }
const VW = 900
const VH = 380
const IW = VW - M.left - M.right
const IH = VH - M.top - M.bottom

const PALETTE = ['#9fb8a9', '#7eb3c9', '#d4a9b8', '#c4b87e', '#9e8fc4', '#7ec4b3', '#c4a07e', '#b37e9e']
const OBS_COLOR = '#7eb3c9'
const PENDING_COLOR = '#c4b87e'

function timeTicks(tMin, tMax, n) {
  const step = (tMax - tMin) / n
  return Array.from({ length: n + 1 }, (_, i) => tMin + i * step)
}

function valueTicks(vMin, vMax, n) {
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

export function TimeseriesChart({ series, observations, rangeMs, onTimePick, pendingTime, yLabel }) {
  const [cursor, setCursor] = useState(null)

  const allT = series.flatMap(s => s.points.map(p => p.t))
  const allV = series.flatMap(s => s.points.map(p => p.v))
  if (!allT.length) return null

  const tMin = Math.min(...allT)
  const tMax = Math.max(...allT)
  const vMin = Math.min(...allV)
  const vMax = Math.max(...allV)
  const vRange = vMax - vMin || 1

  const x = t => M.left + ((t - tMin) / (tMax - tMin || 1)) * IW
  const y = v => M.top + IH - ((v - vMin) / vRange) * IH

  const xTicks = timeTicks(tMin, tMax, 6)
  const yTicks = valueTicks(vMin, vMax, 5)

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
        {yTicks.map(v => (
          <line key={v} x1={M.left} x2={VW - M.right} y1={y(v)} y2={y(v)} className="grid-line" />
        ))}

        {series.map(s => (
          <polyline
            key={s.label}
            points={s.points.map(p => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')}
            fill="none"
            stroke={s.color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            opacity="0.9"
          />
        ))}

        {(observations ?? []).filter(o => inRange(o.observed_at)).map(o => {
          const confirmed = o.note?.includes('WATER') && !o.note?.includes('AUTO')
          return (
            <g key={o.id}>
              <line
                x1={x(o.observed_at)} x2={x(o.observed_at)}
                y1={M.top} y2={M.top + IH}
                stroke={OBS_COLOR} strokeWidth="1.5" strokeDasharray="4 3" opacity="0.8"
              />
              {confirmed && (
                <text x={x(o.observed_at)} y={M.top - 4} textAnchor="middle"
                  fontSize="13" fill={OBS_COLOR} opacity="0.9">★</text>
              )}
            </g>
          )
        })}

        <line x1={M.left} x2={VW - M.right} y1={M.top + IH} y2={M.top + IH} className="axis-line" />
        {xTicks.map(t => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={M.top + IH} y2={M.top + IH + 5} className="axis-line" />
            <text x={x(t)} y={M.top + IH + 18} className="axis-label" textAnchor="middle">
              {fmtTime(t, rangeMs)}
            </text>
          </g>
        ))}

        <line x1={M.left} x2={M.left} y1={M.top} y2={M.top + IH} className="axis-line" />
        {yLabel && (
          <text
            x={14} y={M.top + IH / 2}
            className="axis-label"
            textAnchor="middle"
            transform={`rotate(-90, 14, ${M.top + IH / 2})`}
          >
            {yLabel}
          </text>
        )}
        {yTicks.map(v => (
          <g key={v}>
            <line x1={M.left - 5} x2={M.left} y1={y(v)} y2={y(v)} className="axis-line" />
            <text x={M.left - 9} y={y(v) + 4} className="axis-label" textAnchor="end">
              {Math.round(v)}
            </text>
          </g>
        ))}

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
        <span className="legend-item">
          <span className="legend-dot" style={{ background: OBS_COLOR, borderRadius: 2 }} />
          recorded watering
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: PENDING_COLOR, borderRadius: 2 }} />
          candidate
        </span>
      </div>
    </div>
  )
}

export { PALETTE }
