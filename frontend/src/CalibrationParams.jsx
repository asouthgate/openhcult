export default function CalibrationParams({ calibParams, setCalibParam }) {
  const { offsetMin, widthMin, prior, priorMin, priorMax, estimator, priorWeight, nBurn, nSteps, emaTauMin, systemCapacityMean, systemCapacityStd } = calibParams
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