export default function CalibrationParams({ calibParams, setCalibParam }) {
  const { offsetMin, widthMin, estimator, priorWeight, nBurn, nSteps, emaTauMin, systemCapacityMean, systemCapacityStd, nRecent, withinDays } = calibParams
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
          <td>EMA tau</td>
          <td><input type="number" min="1" step="1" className="param-input" value={emaTauMin} onChange={e => setCalibParam('emaTauMin', e.target.value)} /></td>
          <td>min</td>
        </tr>
        <tr>
          <td>n per sensor (most recent)</td>
          <td><input type="number" min="1" step="1" className="param-input" value={nRecent} onChange={e => setCalibParam('nRecent', e.target.value)} /></td>
          <td></td>
        </tr>
        <tr>
          <td>within</td>
          <td><input type="number" min="0" className="param-input" value={withinDays} onChange={e => setCalibParam('withinDays', e.target.value)} /></td>
          <td>days <span className="param-hint">(empty&nbsp;=&nbsp;all)</span></td>
        </tr>
        <tr>
          <td>estimator</td>
          <td colSpan={2}>
            <div className="range-btns">
              <button className={estimator === 'exponential' ? 'active' : ''} onClick={() => setCalibParam('estimator', 'exponential')}>exponential</button>
              <button className={estimator === 'exp_mcmc' ? 'active' : ''} onClick={() => setCalibParam('estimator', 'exp_mcmc')}>exp MCMC</button>
            </div>
          </td>
        </tr>
        {(estimator === 'exponential' || estimator === 'exp_mcmc') && (
          <tr>
            <td>prior weight</td>
            <td><input type="number" min="0.001" step="0.1" className="param-input" value={priorWeight} onChange={e => setCalibParam('priorWeight', e.target.value)} /></td>
            <td></td>
          </tr>
        )}
        {estimator === 'exp_mcmc' && <>
          <tr>
            <td>burn-in</td>
            <td><input type="number" min="0" step="1" className="param-input" value={nBurn} onChange={e => setCalibParam('nBurn', e.target.value)} /></td>
            <td>steps</td>
          </tr>
          <tr>
            <td>samples</td>
            <td><input type="number" min="1" step="1" className="param-input" value={nSteps} onChange={e => setCalibParam('nSteps', e.target.value)} /></td>
            <td>steps</td>
          </tr>
          <tr>
            <td>sys capacity mean</td>
            <td><input type="number" step="1" className="param-input" value={systemCapacityMean} onChange={e => setCalibParam('systemCapacityMean', e.target.value)} /></td>
            <td>ml</td>
          </tr>
          <tr>
            <td>sys capacity std</td>
            <td><input type="number" step="1" className="param-input" value={systemCapacityStd} onChange={e => setCalibParam('systemCapacityStd', e.target.value)} /></td>
            <td>ml</td>
          </tr>
        </>}
      </tbody>
    </table>
  )
}