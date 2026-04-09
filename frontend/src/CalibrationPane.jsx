import { useState } from 'react'
import CalibrationCurve from './CalibrationCurve'

export default function CalibrationPane({ plantFilter, calibration, calibError, calibLoading }) {
  const [showPct, setShowPct] = useState(false)

  if (!plantFilter) return <div className="empty">Select a plant to view calibration.</div>
  if (calibLoading) return <div>Computing…</div>
  if (calibError) return <div className="error">{calibError}</div>
  if (!calibration) return <div className="empty">No calibration data.</div>

  return (
    <section className="calibration-pane">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <h3 style={{ margin: 0 }}>Calibration: {plantFilter}</h3>
        <div className="range-btns">
          <button className={!showPct ? 'active' : ''} onClick={() => setShowPct(false)}>ml</button>
          <button className={showPct ? 'active' : ''} onClick={() => setShowPct(true)}>%FC</button>
        </div>
      </div>
      <CalibrationCurve calibration={calibration} showPct={showPct} />
    </section>
  )
}
