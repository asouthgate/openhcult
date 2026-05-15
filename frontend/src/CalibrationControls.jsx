import CalibrationParams from './CalibrationParams'

export default function CalibrationControls({ calibParams, setCalibParam, recalculate, calibLoading, plantFilter }) {
  return (
    <div>
      <CalibrationParams calibParams={calibParams} setCalibParam={setCalibParam} />
      <button className="recalc-btn" onClick={recalculate} disabled={calibLoading || !plantFilter}>
        {calibLoading ? 'Computing…' : 'Recalculate'}
      </button>
    </div>
  )
}
