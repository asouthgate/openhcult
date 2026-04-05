import CalibrationCurve from './CalibrationCurve'

export default function CalibrationPane({ plantFilter, calibration, calibError, calibLoading }) {
  if (!plantFilter) return <div className="empty">Select a plant to view calibration.</div>
  if (calibLoading) return <div>Computing…</div>
  if (calibError) return <div className="error">{calibError}</div>
  if (!calibration) return <div className="empty">No calibration data.</div>

  return (
    <section className="calibration-pane">
      <h3>Calibration: {plantFilter}</h3>
      <CalibrationCurve calibration={calibration} />
    </section>
  )
}
