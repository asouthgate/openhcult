export default function CalibrationParams({ calibParams, setCalibParam }) {
  const { offsetMin, widthMin, prior, priorMin, priorMax, priorAlpha, estimator, priorWeight, nBurn, nSteps, xminLow, xminHigh } = calibParams
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
            <td>xmin low</td>
            <td><input type="number" step="1" className="param-input" value={xminLow} onChange={e => setCalibParam('xminLow', e.target.value)} /></td>
            <td>mV</td>
          </tr>
          <tr>
            <td>xmin high</td>
            <td><input type="number" step="1" className="param-input" value={xminHigh} onChange={e => setCalibParam('xminHigh', e.target.value)} /></td>
            <td>mV</td>
          </tr>
        </>}
        <tr>
          <td>prior</td>
          <td colSpan={2}>
            <div className="range-btns">
              <button className={prior === 'calibrated' ? 'active' : ''} onClick={() => setCalibParam('prior', 'calibrated')}>calibrated</button>
              <button className={prior === 'linear' ? 'active' : ''} onClick={() => setCalibParam('prior', 'linear')}>linear</button>
              <button className={prior === 'power' ? 'active' : ''} onClick={() => setCalibParam('prior', 'power')}>power</button>
            </div>
          </td>
        </tr>
        {(prior === 'linear' || prior === 'power') && <>
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
        {prior === 'power' && (
          <tr>
            <td>α</td>
            <td><input type="number" min="0.01" step="0.05" className="param-input" value={priorAlpha} onChange={e => setCalibParam('priorAlpha', e.target.value)} /></td>
            <td></td>
          </tr>
        )}
      </tbody>
    </table>
  )
}
