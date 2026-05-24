import { valueTicks, SvgAxes, M, VW, VH, IW, IH } from './TimeseriesChart'
import { OBS_COLOR, PALETTE } from './theme'

export default function CalibrationCurve({ calibration, showFractional = false }) {
  if (!calibration) return null

  const { curve_x, prior_x, prior_y, mean, ci_low, ci_high, chords_x, chords_dx, chords_dy, mean_at_chord_starts, sensor_chord_labels, per_sensor_curves } = calibration

  const mainX = curve_x ?? prior_x

  const gpRange = Math.max(...mean) - Math.min(...mean)
  const priorScaled = prior_y.map(v => v * gpRange)

  const chordEndYs = chords_x.map((_, i) => mean_at_chord_starts[i] + chords_dy[i])

  const xMin = Math.min(...prior_x, ...mainX, ...chords_x, ...chords_x.map((x, i) => x + chords_dx[i]),
    ...(per_sensor_curves?.flatMap(c => c.x) ?? []))
  const xMax = Math.max(...prior_x, ...mainX, ...chords_x, ...chords_x.map((x, i) => x + chords_dx[i]),
    ...(per_sensor_curves?.flatMap(c => c.x) ?? []))
  const yMin = Math.min(0, ...ci_low, ...chordEndYs,
    ...(per_sensor_curves?.flatMap(c => c.ci_low) ?? []))
  const yMax = Math.max(...ci_high, ...chordEndYs,
    ...(per_sensor_curves?.flatMap(c => c.ci_high) ?? []))

  const scx = v => M.left + ((v - xMin) / (xMax - xMin || 1)) * IW
  const scy = v => M.top + IH - ((v - yMin) / ((yMax - yMin) || 1)) * IH

  const xypts = (xs, ys) => xs.map((xi, i) => `${scx(xi).toFixed(1)},${scy(ys[i]).toFixed(1)}`).join(' ')

  const ciUpperPts = xypts(mainX, ci_high)
  const ciLowerPts = xypts([...mainX].reverse(), [...ci_low].reverse())

  const xTicks = valueTicks(xMin, xMax, 6)
  const yTicks = valueTicks(yMin, yMax, 5)

  const chordColor = i => {
    if (!sensor_chord_labels) return OBS_COLOR
    return PALETTE[sensor_chord_labels[i] % PALETTE.length]
  }

  const hasMultiSensor = per_sensor_curves?.length > 0

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

        {hasMultiSensor && per_sensor_curves.map((c, j) => {
          const color = PALETTE[j % PALETTE.length]
          const ciUpperPtsJ = xypts(c.x, c.ci_high)
          const ciLowerPtsJ = xypts([...c.x].reverse(), [...c.ci_low].reverse())
          return [
            <polygon key={`ci-fill-${j}`} points={`${ciUpperPtsJ} ${ciLowerPtsJ}`} fill={color} opacity="0.15" />,
            <polyline key={`ci-hi-${j}`} points={ciUpperPtsJ} fill="none" stroke={color} strokeWidth="1" opacity="0.4" />,
            <polyline key={`ci-lo-${j}`} points={ciLowerPtsJ} fill="none" stroke={color} strokeWidth="1" opacity="0.4" />,
          ]
        })}

        <polyline
          points={xypts(prior_x, priorScaled)}
          fill="none" stroke="#ffc61c" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7"
        />

        {hasMultiSensor && per_sensor_curves.map((c, j) => (
          <polyline
            key={j}
            points={xypts(c.x, c.mean)}
            fill="none" stroke={PALETTE[j % PALETTE.length]} strokeWidth="1.5" opacity="0.8"
          />
        ))}

        <polyline
          points={xypts(mainX, mean)}
          fill="none" stroke="#d0fffc" strokeWidth={hasMultiSensor ? 2.5 : 2} strokeLinejoin="round"
        />

        {chords_x.map((xi, i) => {
          const color = chordColor(i)
          const x0 = scx(xi).toFixed(1)
          const x1 = scx(xi + chords_dx[i]).toFixed(1)
          const y0 = scy(mean_at_chord_starts[i]).toFixed(1)
          const y1 = scy(mean_at_chord_starts[i] + chords_dy[i]).toFixed(1)
          return (
            <g key={i}>
              <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={color} strokeWidth="1.5" opacity="0.5" />
              <circle cx={x0} cy={y0} r="3" fill={color} opacity="0.5" />
              <circle cx={x1} cy={y1} r="3" fill={color} opacity="0.5" />
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
        <span className="legend-item"><span className="legend-dot" style={{ background: '#d0fffc' }} />{hasMultiSensor ? 'Joint diagonal' : 'GP mean'}</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: OBS_COLOR, opacity: 0.3 }} />95% CI</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: '#ffc61c' }} />prior</span>
        {hasMultiSensor && calibration.active_sensors?.map((s, j) => (
          <span key={j} className="legend-item"><span className="legend-dot" style={{ background: PALETTE[j % PALETTE.length] }} />{s.device_address} / {s.sensor}</span>
        ))}
        {!hasMultiSensor && <span className="legend-item"><span className="legend-dot" style={{ background: OBS_COLOR }} />chords</span>}
      </div>
    </div>
  )
}