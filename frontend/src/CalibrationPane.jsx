import { useState } from 'react'
import CalibrationCurve, { ScatterPlot } from './CalibrationCurve'
import CalibrationParams from './CalibrationParams'

export default function CalibrationPane({
  plantFilter, sensorFilter, calibration, calibError, calibLoading,
  offsetMin, setOffsetMin, widthMin, setWidthMin,
  gpStdMl, setGpStdMl, scalePriorMean, setScalePriorMean, scalePriorStd, setScalePriorStd,
}) {
  const [showPct, setShowPct] = useState(false)

  if (!plantFilter) return <div className="empty">Select a plant to view calibration.</div>
  if (calibLoading) return <div>Computing…</div>
  if (calibError) return <div className="error">{calibError}</div>
  if (!calibration) return <div className="empty">No calibration data.</div>

  const { chords_dx, chords_dy, scale, nlml } = calibration

  return (
    <section className="calibration-pane">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <h3 style={{ margin: 0 }}>Calibration: {plantFilter}{sensorFilter ? ` / ${sensorFilter}` : ''}</h3>
        <div className="range-btns">
          <button className={!showPct ? 'active' : ''} onClick={() => setShowPct(false)}>ml</button>
          <button className={showPct ? 'active' : ''} onClick={() => setShowPct(true)}>%FC</button>
        </div>
        {nlml != null && (
          <span style={{ fontSize: 12, opacity: 0.7 }}>NLML: {nlml.toFixed(2)}</span>
        )}
      </div>
      <CalibrationCurve calibration={calibration} showPct={showPct} />
      <div style={{ display: 'flex', gap: 24, marginTop: 16, alignItems: 'flex-start' }}>
        <ScatterPlot
          dx={chords_dx}
          dy={chords_dy.map(v => showPct ? (v / scale) * 100 : v)}
          xLabel="Δsensor"
          yLabel={showPct ? 'Δ%FC' : 'Δml'}
        />
        <CalibrationParams
          offsetMin={offsetMin} setOffsetMin={setOffsetMin}
          widthMin={widthMin} setWidthMin={setWidthMin}
          gpStdMl={gpStdMl} setGpStdMl={setGpStdMl}
          scalePriorMean={scalePriorMean} setScalePriorMean={setScalePriorMean}
          scalePriorStd={scalePriorStd} setScalePriorStd={setScalePriorStd}
        />
      </div>
    </section>
  )
}
