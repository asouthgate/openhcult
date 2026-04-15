export default function CalibrationParams({ calibParams, setCalibParam }) {
  const { offsetMin, widthMin, gpStdMl, scalePriorMean, scalePriorStd } = calibParams
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
      </tbody>
    </table>
  )
}
