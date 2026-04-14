export default function CalibrationParams({
  offsetMin, setOffsetMin, widthMin, setWidthMin,
  gpStdMl, setGpStdMl, scalePriorMean, setScalePriorMean, scalePriorStd, setScalePriorStd,
}) {
  return (
    <table className="param-table">
      <tbody>
        <tr>
          <td>offset</td>
          <td><input type="number" min="0" value={offsetMin} onChange={e => setOffsetMin(e.target.value)} style={{ width: 52 }} /></td>
          <td>min</td>
        </tr>
        <tr>
          <td>width</td>
          <td><input type="number" min="1" value={widthMin} onChange={e => setWidthMin(e.target.value)} style={{ width: 52 }} /></td>
          <td>min</td>
        </tr>
        <tr>
          <td>GP std</td>
          <td><input type="number" min="0.1" step="0.1" value={gpStdMl} onChange={e => setGpStdMl(e.target.value)} style={{ width: 52 }} /></td>
          <td>ml</td>
        </tr>
        <tr>
          <td>scale prior μ</td>
          <td><input type="number" min="0" step="10" value={scalePriorMean} onChange={e => setScalePriorMean(e.target.value)} placeholder="off" style={{ width: 60 }} /></td>
          <td>ml</td>
        </tr>
        <tr>
          <td>scale prior σ</td>
          <td><input type="number" min="0" step="10" value={scalePriorStd} onChange={e => setScalePriorStd(e.target.value)} placeholder="off" style={{ width: 60 }} /></td>
          <td>ml</td>
        </tr>
      </tbody>
    </table>
  )
}
