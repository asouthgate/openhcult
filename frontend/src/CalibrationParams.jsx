export default function CalibrationParams({ calibParams, setCalibParam }) {
  const { offsetMin, widthMin, gpStdMl, scalePriorMean, scalePriorStd, prior, priorMin, priorMax } = calibParams
  return (
    <table className="param-table">
      <tbody>
        <tr>
          <td>offset</td>
          <td><input type="number" min="0" className="param-input" value={offsetMin} onChange={e => setCalibParam('offsetMin', e.target.value)} /></td>
          <td>min</td>
        </tr>
        <tr>
          <td>width</td>
          <td><input type="number" min="1" className="param-input" value={widthMin} onChange={e => setCalibParam('widthMin', e.target.value)} /></td>
          <td>min</td>
        </tr>
        <tr>
          <td>GP std</td>
          <td><input type="number" min="0.1" step="0.1" className="param-input" value={gpStdMl} onChange={e => setCalibParam('gpStdMl', e.target.value)} /></td>
          <td>ml</td>
        </tr>
        <tr>
          <td>scale prior μ</td>
          <td><input type="number" min="0" step="10" className="param-input-wide" value={scalePriorMean} onChange={e => setCalibParam('scalePriorMean', e.target.value)} placeholder="off" /></td>
          <td>ml</td>
        </tr>
        <tr>
          <td>scale prior σ</td>
          <td><input type="number" min="0" step="10" className="param-input-wide" value={scalePriorStd} onChange={e => setCalibParam('scalePriorStd', e.target.value)} placeholder="off" /></td>
          <td>ml</td>
        </tr>
        <tr>
          <td>prior</td>
          <td colSpan={2}>
            <div className="range-btns">
              <button className={prior === 'calibrated' ? 'active' : ''} onClick={() => setCalibParam('prior', 'calibrated')}>calibrated</button>
              <button className={prior === 'linear' ? 'active' : ''} onClick={() => setCalibParam('prior', 'linear')}>linear</button>
            </div>
          </td>
        </tr>
        {prior === 'linear' && <>
          <tr>
            <td>prior min</td>
            <td><input type="number" className="param-input" value={priorMin} onChange={e => setCalibParam('priorMin', e.target.value)} placeholder="mV" /></td>
            <td>mV</td>
          </tr>
          <tr>
            <td>prior max</td>
            <td><input type="number" className="param-input" value={priorMax} onChange={e => setCalibParam('priorMax', e.target.value)} placeholder="mV" /></td>
            <td>mV</td>
          </tr>
        </>}
      </tbody>
    </table>
  )
}
