export function buildSwcTimeseriesParams(plantFilter, calibParams, rangeHours) {
  const { offsetMin, widthMin, prior, priorMin, priorMax, priorWeight, nBurn, nSteps, xminMu, xminSigma, xminHigh } = calibParams
  const endMs = Date.now()
  const params = new URLSearchParams({
    plant: plantFilter,
    offset_ms: Number(offsetMin) * 60 * 1000,
    width_ms: Number(widthMin) * 60 * 1000,
    start_ms: endMs - rangeHours * 3600 * 1000,
    end_ms: endMs,
  })
  if (priorMin !== '') params.set('prior_min', priorMin)
  if (priorMax !== '') params.set('prior_max', priorMax)
  if (prior !== 'calibrated') params.set('prior', prior)
  if (priorWeight !== '') params.set('prior_weight', priorWeight)
  if (nBurn !== '') params.set('n_burn', nBurn)
  if (nSteps !== '') params.set('n_steps', nSteps)
  if (xminMu !== '') params.set('xmin_mu', xminMu)
  if (xminSigma !== '') params.set('xmin_sigma', xminSigma)
  if (xminHigh !== '') params.set('xmin_high', xminHigh)
  return params
}

export function buildDryingRateParams(plantFilter, sensorFilter, calibParams, rangeHours, combined = false) {
  const { offsetMin, widthMin, prior, priorMin, priorMax, estimator, priorWeight, nBurn, nSteps, xminMu, xminSigma, xminHigh, emaTauMin } = calibParams
  const endMs = Date.now()
  const params = new URLSearchParams({
    plant: plantFilter,
    offset_ms: Number(offsetMin) * 60 * 1000,
    width_ms: Number(widthMin) * 60 * 1000,
    start_utc: new Date(endMs - rangeHours * 3600 * 1000).toISOString(),
    end_utc: new Date(endMs).toISOString(),
  })
  if (combined) params.set('combined', 'true')
  if (sensorFilter) {
    const sep = sensorFilter.lastIndexOf(':')
    params.set('sensor', sensorFilter.slice(sep + 1))
    params.set('device_address', sensorFilter.slice(0, sep))
  }
  if (priorMin !== '') params.set('prior_min', priorMin)
  if (priorMax !== '') params.set('prior_max', priorMax)
  if (prior !== 'calibrated') params.set('prior', prior)
  if (estimator !== 'exp_mcmc') params.set('estimator', estimator)
  if (priorWeight !== '') params.set('prior_weight', priorWeight)
  if (estimator === 'exp_mcmc') {
    if (nBurn !== '') params.set('n_burn', nBurn)
    if (nSteps !== '') params.set('n_steps', nSteps)
    if (xminMu !== '') params.set('xmin_mu', xminMu)
    if (xminSigma !== '') params.set('xmin_sigma', xminSigma)
    if (xminHigh !== '') params.set('xmin_high', xminHigh)
  }
  if (emaTauMin !== '') params.set('ema_tau_min', emaTauMin)
  return params
}

export function buildWaterCalibrationParams(plantFilter, sensorFilter, calibParams) {
  const { offsetMin, widthMin, prior, priorMin, priorMax, estimator, priorWeight, nBurn, nSteps, xminMu, xminSigma, xminHigh } = calibParams
  const params = new URLSearchParams({
    plant: plantFilter,
    offset_ms: Number(offsetMin) * 60 * 1000,
    width_ms: Number(widthMin) * 60 * 1000,
  })
  if (sensorFilter) {
    const sep = sensorFilter.lastIndexOf(':')
    params.set('sensor', sensorFilter.slice(sep + 1))
    params.set('device_address', sensorFilter.slice(0, sep))
  }
  if (estimator !== 'exp_mcmc') params.set('estimator', estimator)
  if (estimator === 'exponential' && priorWeight !== '') params.set('prior_weight', priorWeight)
  if (estimator === 'exp_mcmc') {
    if (priorWeight !== '') params.set('prior_weight', priorWeight)
    if (nBurn !== '') params.set('n_burn', nBurn)
    if (nSteps !== '') params.set('n_steps', nSteps)
    if (xminMu !== '') params.set('xmin_mu', xminMu)
    if (xminSigma !== '') params.set('xmin_sigma', xminSigma)
    if (xminHigh !== '') params.set('xmin_high', xminHigh)
  }
  if (prior !== 'calibrated') params.set('prior', prior)
  if (prior === 'linear') {
    if (priorMin !== '') params.set('prior_min', priorMin)
    if (priorMax !== '') params.set('prior_max', priorMax)
  }
  if (estimator === 'exponential' || estimator === 'exp_mcmc') {
    if (priorMin !== '' && !params.has('prior_min')) params.set('prior_min', priorMin)
    if (priorMax !== '' && !params.has('prior_max')) params.set('prior_max', priorMax)
  }
  return params
}
