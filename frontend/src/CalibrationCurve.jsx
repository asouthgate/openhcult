import { valueTicks } from './TimeseriesChart'

const M = { top: 20, right: 24, bottom: 52, left: 60 }
const VW = 900
const VH = 380
const IW = VW - M.left - M.right
const IH = VH - M.top - M.bottom

const SM = { top: 20, right: 24, bottom: 52, left: 60 }
const SVW = 340
const SVH = 280
const SIW = SVW - SM.left - SM.right
const SIH = SVH - SM.top - SM.bottom

function ScatterPlot({ dx, dy, xLabel, yLabel }) {
  if (!dx?.length) return null
  const xMin = Math.min(...dx), xMax = Math.max(...dx)
  const yMin = Math.min(...dy), yMax = Math.max(...dy)
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1
  const sx = v => SM.left + ((v - xMin) / xRange) * SIW
  const sy = v => SM.top + SIH - ((v - yMin) / yRange) * SIH
  const xTicks = valueTicks(xMin, xMax, 4)
  const yTicks = valueTicks(yMin, yMax, 4)
  const x0 = sx(Math.max(xMin, Math.min(xMax, 0)))
  const y0 = sy(Math.max(yMin, Math.min(yMax, 0)))
  return (
    <svg viewBox={`0 0 ${SVW} ${SVH}`} className="chart-svg" style={{ maxWidth: SVW }}>
      {yTicks.map(v => <line key={v} x1={SM.left} x2={SVW - SM.right} y1={sy(v)} y2={sy(v)} className="grid-line" />)}
      <line x1={x0} x2={x0} y1={SM.top} y2={SM.top + SIH} className="grid-line" />
      <line x1={SM.left} x2={SVW - SM.right} y1={y0} y2={y0} className="grid-line" />
      {dx.map((dxi, i) => (
        <g key={i}>
          <circle cx={sx(dxi)} cy={sy(dy[i])} r="4" fill="#7eb3c9" opacity="0.7" />
          <text x={sx(dxi) + 5} y={sy(dy[i]) - 4} fontSize="9" fill="#7eb3c9" opacity="0.7">{i}</text>
        </g>
      ))}
      <line x1={SM.left} x2={SVW - SM.right} y1={SM.top + SIH} y2={SM.top + SIH} className="axis-line" />
      <line x1={SM.left} x2={SM.left} y1={SM.top} y2={SM.top + SIH} className="axis-line" />
      {xTicks.map(v => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={SM.top + SIH} y2={SM.top + SIH + 5} className="axis-line" />
          <text x={sx(v)} y={SM.top + SIH + 18} className="axis-label" textAnchor="middle">{Math.round(v)}</text>
        </g>
      ))}
      {yTicks.map(v => (
        <g key={v}>
          <line x1={SM.left - 5} x2={SM.left} y1={sy(v)} y2={sy(v)} className="axis-line" />
          <text x={SM.left - 9} y={sy(v) + 4} className="axis-label" textAnchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      <text x={SM.left + SIW / 2} y={SVH - 6} className="axis-label" textAnchor="middle">{xLabel}</text>
      <text x={10} y={SM.top + SIH / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90,10,${SM.top + SIH / 2})`}>{yLabel}</text>
    </svg>
  )
}

export default function CalibrationCurve({ calibration, showPct = false }) {
  if (!calibration) return null

  const { prior_x, prior_y, mean, std, anchors_x, anchors_y, chords_x, chords_dx, chords_dy, mean_at_chord_starts, scale } = calibration

  const toY = v => showPct ? (v / scale) * 100 : v

  const ref = mean[mean.length - 1]
  const meanNorm = mean.map(v => toY(v - ref))
  const ciLo = mean.map((v, i) => toY(v - 1.96 * std[i] - ref))
  const ciHi = mean.map((v, i) => toY(v + 1.96 * std[i] - ref))
  const gpRange = Math.max(...meanNorm) - Math.min(...meanNorm)
  const priorScaled = prior_y.map(v => v * gpRange)

  const chordEndYs = chords_x.map((_, i) => toY(mean_at_chord_starts[i] - ref + chords_dy[i]))

  const xMin = Math.min(...prior_x)
  const xMax = Math.max(...prior_x)
  const yMin = Math.min(0, ...ciLo, ...chordEndYs)
  const yMax = Math.max(...ciHi, ...chordEndYs)

  const scx = v => M.left + ((v - xMin) / (xMax - xMin || 1)) * IW
  const scy = v => M.top + IH - ((v - yMin) / ((yMax - yMin) || 1)) * IH

  const ciUpperPts = prior_x.map((xi, i) => `${scx(xi).toFixed(1)},${scy(ciHi[i]).toFixed(1)}`).join(' ')
  const ciLowerPts = [...prior_x].reverse().map((xi, i) => `${scx(xi).toFixed(1)},${scy(ciLo[prior_x.length - 1 - i]).toFixed(1)}`).join(' ')

  const xTicks = valueTicks(xMin, xMax, 6)
  const yTicks = valueTicks(yMin, yMax, 5)

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="chart-svg">
        {yTicks.map(v => (
          <line key={v} x1={M.left} x2={VW - M.right} y1={scy(v)} y2={scy(v)} className="grid-line" />
        ))}

        <polygon points={`${ciUpperPts} ${ciLowerPts}`} fill="#7eb3c9" opacity="0.35" />
        <polyline points={ciUpperPts} fill="none" stroke="#7eb3c9" strokeWidth="1" opacity="0.6" />
        <polyline points={[...prior_x].reverse().map((xi, i) => `${scx(xi).toFixed(1)},${scy(ciLo[prior_x.length - 1 - i]).toFixed(1)}`).join(' ')} fill="none" stroke="#7eb3c9" strokeWidth="1" opacity="0.6" />

        <polyline
          points={prior_x.map((xi, i) => `${scx(xi).toFixed(1)},${scy(priorScaled[i]).toFixed(1)}`).join(' ')}
          fill="none" stroke="#c4b87e" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7"
        />

        <polyline
          points={prior_x.map((xi, i) => `${scx(xi).toFixed(1)},${scy(meanNorm[i]).toFixed(1)}`).join(' ')}
          fill="none" stroke="#9fb8a9" strokeWidth="2" strokeLinejoin="round"
        />

        {chords_x.map((xi, i) => {
          const x0 = scx(xi).toFixed(1)
          const x1 = scx(xi + chords_dx[i]).toFixed(1)
          const y0 = scy(toY(mean_at_chord_starts[i] - ref)).toFixed(1)
          const y1 = scy(toY(mean_at_chord_starts[i] - ref + chords_dy[i])).toFixed(1)
          return (
            <g key={i}>
              <line x1={x0} y1={y0} x2={x1} y2={y1} stroke="#7eb3c9" strokeWidth="1.5" opacity="0.5" />
              <circle cx={x0} cy={y0} r="3" fill="#7eb3c9" opacity="0.5" />
              <circle cx={x1} cy={y1} r="3" fill="#7eb3c9" opacity="0.5" />
            </g>
          )
        })}

        {anchors_x.map((xi, i) => (
          <circle key={i} cx={scx(xi)} cy={scy(anchors_y[i])} r="5" fill="#d4a9b8" />
        ))}

        <line x1={M.left} x2={VW - M.right} y1={M.top + IH} y2={M.top + IH} className="axis-line" />
        <line x1={M.left} x2={M.left} y1={M.top} y2={M.top + IH} className="axis-line" />

        {xTicks.map(v => (
          <g key={v}>
            <line x1={scx(v)} x2={scx(v)} y1={M.top + IH} y2={M.top + IH + 5} className="axis-line" />
            <text x={scx(v)} y={M.top + IH + 18} className="axis-label" textAnchor="middle">{Math.round(v)}</text>
          </g>
        ))}

        {yTicks.map(v => (
          <g key={v}>
            <line x1={M.left - 5} x2={M.left} y1={scy(v)} y2={scy(v)} className="axis-line" />
            <text x={M.left - 9} y={scy(v) + 4} className="axis-label" textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}

        <text x={M.left + IW / 2} y={VH - 6} className="axis-label" textAnchor="middle">sensor reading</text>
        <text x={14} y={M.top + IH / 2} className="axis-label" textAnchor="middle"
          transform={`rotate(-90, 14, ${M.top + IH / 2})`}>{showPct ? 'SWC (%FC)' : 'SWC (ml)'}</text>
      </svg>

      <div className="legend">
        <span className="legend-item"><span className="legend-dot" style={{ background: '#9fb8a9' }} />GP mean</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: '#7eb3c9', opacity: 0.3 }} />95% CI</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: '#c4b87e' }} />prior</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: '#7eb3c9' }} />chords</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: '#d4a9b8' }} />anchor</span>
      </div>
      <div style={{ marginTop: 16 }}>
        <ScatterPlot
          dx={chords_dx}
          dy={chords_dy.map(v => showPct ? (v / scale) * 100 : v)}
          xLabel="Δsensor"
          yLabel={showPct ? 'Δ%FC' : 'Δml'}
        />
      </div>
    </div>
  )
}
