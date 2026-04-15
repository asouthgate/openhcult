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

  const { chords_dx, chords_dy, chord_times, scale, nlml } = calibration ?? {}

  return (
    <section className="calibration-pane">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
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
      {calibLoading && <div>Computing…</div>}
      {calibError && <div className="error">{calibError}</div>}
      {calibration && <>
        <CalibrationCurve calibration={calibration} showPct={showPct} />
        {chord_times?.length > 0 && (
          <details style={{ marginTop: 16 }}>
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
        <div style={{ display: 'flex', gap: 24, marginTop: 16, alignItems: 'flex-start' }}>
          <ScatterPlot
            dx={chords_dx}
            dy={chords_dy.map(v => toWater(v, scale, showPct))}
            xLabel="Δsensor"
            yLabel={showPct ? 'Δ%FC' : 'Δml'}
          />
        </div>
      </>}
      <div style={{ marginTop: 16 }}>
        <CalibrationParams calibParams={calibParams} setCalibParam={setCalibParam} />
      </div>
    </section>
  )
}
