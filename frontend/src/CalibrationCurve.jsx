import { valueTicks, SvgAxes, M, VW, VH, IW, IH } from './TimeseriesChart'
import { OBS_COLOR } from './theme'

export default function CalibrationCurve({ calibration, showFractional = false }) {
  if (!calibration) return null

  const { prior_x, prior_y, mean, ci_low, ci_high, chords_x, chords_dx, chords_dy, mean_at_chord_starts, scale } = calibration

  const ref = showFractional ? 0 : mean[mean.length - 1]
  const meanNorm = mean.map(v => v - ref)
  const ciLo = ci_low ? ci_low.map(v => v - ref) : mean.map((v, i) => v - 1.96 * (calibration.std?.[i] ?? 0) - ref)
  const ciHi = ci_high ? ci_high.map(v => v - ref) : mean.map((v, i) => v + 1.96 * (calibration.std?.[i] ?? 0) - ref)
  const gpRange = Math.max(...meanNorm) - Math.min(...meanNorm)
  const priorScaled = prior_y.map(v => v * gpRange)

  const chordEndYs = chords_x.map((_, i) => mean_at_chord_starts[i] - ref + chords_dy[i])

  const xMin = Math.min(...prior_x, ...chords_x, ...chords_x.map((x, i) => x + chords_dx[i]))
  const xMax = Math.max(...prior_x, ...chords_x, ...chords_x.map((x, i) => x + chords_dx[i]))
  const yMin = Math.min(0, ...ciLo, ...chordEndYs)
  const yMax = Math.max(...ciHi, ...chordEndYs)

  const scx = v => M.left + ((v - xMin) / (xMax - xMin || 1)) * IW
  const scy = v => M.top + IH - ((v - yMin) / ((yMax - yMin) || 1)) * IH

  const xypts = (xs, ys) => xs.map((xi, i) => `${scx(xi).toFixed(1)},${scy(ys[i]).toFixed(1)}`).join(' ')

  const ciUpperPts = xypts(prior_x, ciHi)
  const ciLowerPts = xypts([...prior_x].reverse(), [...ciLo].reverse())

  const xTicks = valueTicks(xMin, xMax, 6)
  const yTicks = valueTicks(yMin, yMax, 5)

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="chart-svg">
        <rect x={M.left} y={M.top} width={IW} height={IH} fill="#191e2b" />
        {yTicks.map(v => (
          <line key={v} x1={M.left} x2={VW - M.right} y1={scy(v)} y2={scy(v)} className="grid-line" />
        ))}
        {xTicks.map(v => (
          <line key={`x${v}`} x1={scx(v)} x2={scx(v)} y1={M.top} y2={M.top + IH} className="grid-line" />
        ))}

        <polygon points={`${ciUpperPts} ${ciLowerPts}`} fill={OBS_COLOR} opacity="0.35" />
        <polyline points={ciUpperPts} fill="none" stroke={OBS_COLOR} strokeWidth="1" opacity="0.6" />
        <polyline points={ciLowerPts} fill="none" stroke={OBS_COLOR} strokeWidth="1" opacity="0.6" />

        <polyline
          points={xypts(prior_x, priorScaled)}
          fill="none" stroke="#ffc61c" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7"
        />

        <polyline
          points={xypts(prior_x, meanNorm)}
          fill="none" stroke="#d0fffc" strokeWidth="2" strokeLinejoin="round"
        />

        {chords_x.map((xi, i) => {
          const x0 = scx(xi).toFixed(1)
          const x1 = scx(xi + chords_dx[i]).toFixed(1)
          const y0 = scy(mean_at_chord_starts[i] - ref).toFixed(1)
          const y1 = scy(mean_at_chord_starts[i] - ref + chords_dy[i]).toFixed(1)
          return (
            <g key={i}>
              <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={OBS_COLOR} strokeWidth="1.5" opacity="0.5" />
              <circle cx={x0} cy={y0} r="3" fill={OBS_COLOR} opacity="0.5" />
              <circle cx={x1} cy={y1} r="3" fill={OBS_COLOR} opacity="0.5" />
            </g>
          )
        })}

        <SvgAxes
          xTicks={xTicks} yTicks={yTicks}
          x={scx} y={scy}
          m={M} vw={VW} iw={IW} ih={IH}
          formatX={v => Math.round(v)} formatY={v => v.toFixed(showFractional ? 3 : 1)}
          xLabel="sensor reading"
          yLabel={showFractional ? 'Fractional content' : 'SWC (ml)'}
        />
      </svg>

      <div className="legend">
        <span className="legend-item"><span className="legend-dot" style={{ background: '#d0fffc' }} />GP mean</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: OBS_COLOR, opacity: 0.3 }} />95% CI</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: '#ffc61c' }} />prior</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: OBS_COLOR }} />chords</span>
      </div>
    </div>
  )
}
