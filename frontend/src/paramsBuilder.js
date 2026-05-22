export function buildSwcTimeseriesParams(plantFilter, calibParams, rangeHours, returnFractional = false) {
  const { offsetMin, widthMin, priorWeight, nBurn, nSteps, systemCapacityMean, systemCapacityStd, nRecent, withinDays } = calibParams
  const endMs = Date.now()
  const params = new URLSearchParams({
    plant: plantFilter,
    offset_ms: Number(offsetMin) * 60 * 1000,
    width_ms: Number(widthMin) * 60 * 1000,
    start_ms: endMs - rangeHours * 3600 * 1000,
    end_ms: endMs,
  })
  if (priorWeight !== '') params.set('prior_weight', priorWeight)
  if (nBurn !== '') params.set('n_burn', nBurn)
  if (nSteps !== '') params.set('n_steps', nSteps)
  if (systemCapacityMean !== '') params.set('system_capacity_mean', systemCapacityMean)
  if (systemCapacityStd !== '') params.set('system_capacity_std', systemCapacityStd)
  if (returnFractional) params.set('return_fractional', 'true')
  if (nRecent !== '') params.set('n_recent', nRecent)
  if (withinDays !== '') params.set('within_ms', String(Number(withinDays) * 24 * 3600 * 1000))
  return params
}

export function buildDryingRateParams(plantFilter, sensorFilter, calibParams, rangeHours, combined = false, returnFractional = false) {
  const { offsetMin, widthMin, estimator, priorWeight, nBurn, nSteps, emaTauMin, systemCapacityMean, systemCapacityStd } = calibParams
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
  if (estimator !== 'exp_mcmc') params.set('estimator', estimator)
  if (priorWeight !== '') params.set('prior_weight', priorWeight)
  if (estimator === 'exp_mcmc') {
    if (nBurn !== '') params.set('n_burn', nBurn)
    if (nSteps !== '') params.set('n_steps', nSteps)
    if (systemCapacityMean !== '') params.set('system_capacity_mean', systemCapacityMean)
    if (systemCapacityStd !== '') params.set('system_capacity_std', systemCapacityStd)
    if (returnFractional) params.set('return_fractional', 'true')
  }
  if (emaTauMin !== '') params.set('ema_tau_min', emaTauMin)
  return params
}

export function buildWaterCalibrationParams(plantFilter, sensorFilter, calibParams, returnFractional = false) {
  const { offsetMin, widthMin, estimator, priorWeight, nBurn, nSteps, systemCapacityMean, systemCapacityStd, nRecent, withinDays } = calibParams
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
    if (systemCapacityMean !== '') params.set('system_capacity_mean', systemCapacityMean)
    if (systemCapacityStd !== '') params.set('system_capacity_std', systemCapacityStd)
    if (returnFractional) params.set('return_fractional', 'true')
  }
  if (nRecent !== '') params.set('n_recent', nRecent)
  if (withinDays !== '') params.set('within_ms', String(Number(withinDays) * 24 * 3600 * 1000))
  return params
}
