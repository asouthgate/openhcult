import { useState } from 'react'
import { sensorPart, toWater } from './utils'
import CalibrationCurve from './CalibrationCurve'
import ScatterPlot from './ScatterPlot'
import CalibrationParams from './CalibrationParams'

export default function CalibrationPane({
  plantFilter, sensorFilter, calibration, calibError, calibLoading, calibParams, setCalibParam,
}) {
  const [showPct, setShowPct] = useState(false)

  if (!plantFilter) return <div className="empty">Select a plant to view calibration.</div>

  const { chords_dx, chords_dy, chord_times, scale, nlml, mean, prior_x, ci_low, ci_high } = calibration ?? {}

  const showPctLabel = showPct ? ' %FC' : ' ml'
  const swcStats = calibration ? (() => {
    const toV = v => toWater(v, scale, showPct)
    const ref = mean[mean.length - 1]
    const estMin = toV(Math.min(...mean) - ref)
    const estMax = toV(Math.max(...mean) - ref)
    const lo = ci_low ? toV(Math.min(...ci_low) - ref) : null
    const hi = ci_high ? toV(Math.max(...ci_high) - ref) : null
    return { estMin, estMax, lo, hi }
  })() : null

  return (
    <section className="pane-grid">
      <div className="full" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <h3 style={{ margin: 0 }}>Calibration: {plantFilter}{sensorFilter ? ` / ${sensorPart(sensorFilter)}` : ''}</h3>
        {!calibLoading && calibration && (
          <div className="range-btns">
            <button className={!showPct ? 'active' : ''} onClick={() => setShowPct(false)}>ml</button>
            <button className={showPct ? 'active' : ''} onClick={() => setShowPct(true)}>%FC</button>
          </div>
        )}
        {nlml != null && (
          <span style={{ fontSize: 12, opacity: 0.7 }}>NLML: {nlml.toFixed(2)}</span>
        )}
      </div>
      {calibLoading && <div className="full loading"><span className="spinner" />Computing…</div>}
      {calibError && <div className="full error">{calibError}</div>}
      {calibration && <>
        <div className="full">
          <CalibrationCurve calibration={calibration} showPct={showPct} />
        </div>
        {chord_times?.length > 0 && (() => {
          const dy = chords_dy.map(v => toWater(v, scale, showPct))
          const stats = [
            ['Events', chord_times.length],
            ['Earliest', new Date(Math.min(...chord_times)).toLocaleString()],
            ['Latest', new Date(Math.max(...chord_times)).toLocaleString()],
            ['Δsensor min', Math.min(...chords_dx).toFixed(1) + ' mV'],
            ['Δsensor max', Math.max(...chords_dx).toFixed(1) + ' mV'],
            ['Dose min', Math.min(...dy).toFixed(1) + showPctLabel],
            ['Dose max', Math.max(...dy).toFixed(1) + showPctLabel],
            ['Dose mean', (dy.reduce((a, b) => a + b, 0) / dy.length).toFixed(1) + showPctLabel],
            ['SWC range', swcStats ? `${swcStats.estMin.toFixed(1)}–${swcStats.estMax.toFixed(1)}${showPctLabel}` : '—'],
            ...(swcStats?.lo != null ? [['95% CI', `${swcStats.lo.toFixed(1)}–${swcStats.hi.toFixed(1)}${showPctLabel}`]] : []),
          ]
          return (
            <table className="obs-table full">
              <tbody>
                {stats.map(([label, value]) => (
                  <tr key={label}><td style={{ opacity: 0.6 }}>{label}</td><td>{value}</td></tr>
                ))}
              </tbody>
            </table>
          )
        })()}
        {chord_times?.length > 0 && (
          <details className="full">
            <summary style={{ cursor: 'pointer', fontSize: 12, opacity: 0.7 }}>Chord events ({chord_times.length})</summary>
            <table className="obs-table" style={{ marginTop: 8 }}>
              <thead><tr><th>Watering time</th><th>Δsensor (mV)</th><th>Dose (ml)</th></tr></thead>
              <tbody>
                {chord_times.map((t, i) => (
                  <tr key={t}>
                    <td>{new Date(t).toLocaleString()}</td>
                    <td>{chords_dx[i]?.toFixed(1)}</td>
                    <td>{chords_dy[i]?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}
      </>}
      {calibration && (
        <div>
          <ScatterPlot
            dx={chords_dx}
            dy={chords_dy.map(v => toWater(v, scale, showPct))}
            xLabel="Δsensor"
            yLabel={showPct ? 'Δ%FC' : 'Δml'}
          />
        </div>
      )}
      <div>
        <CalibrationParams calibParams={calibParams} setCalibParam={setCalibParam} />
      </div>
    </section>
  )
}
