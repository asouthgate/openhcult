import { describe, it, expect } from 'vitest'
import { buildSwcTimeseriesParams, buildDryingRateParams, buildWaterCalibrationParams } from './paramsBuilder'

const defaultCalibParams = {
  offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
  estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
  xminMu: '850', xminSigma: '75', xminHigh: '1100', emaTauMin: '60',
}

describe('buildSwcTimeseriesParams', () => {
  it('excludes prior param when calibrated', () => {
    const params = buildSwcTimeseriesParams('plant1', defaultCalibParams, 48)
    expect(params.has('prior')).toBe(false)
  })

  it('includes prior param when not calibrated', () => {
    const params = buildSwcTimeseriesParams('plant1', { ...defaultCalibParams, prior: 'linear' }, 48)
    expect(params.get('prior')).toBe('linear')
  })

  it('excludes empty optional params', () => {
    const params = buildSwcTimeseriesParams('plant1', { ...defaultCalibParams, priorMin: '', priorMax: '' }, 48)
    expect(params.has('prior_min')).toBe(false)
  })
})

describe('buildDryingRateParams', () => {
  it('parses sensor filter into separate params', () => {
    const params = buildDryingRateParams('plant1', 'abc123:temperature', defaultCalibParams, 48)
    expect(params.get('sensor')).toBe('temperature')
    expect(params.get('device_address')).toBe('abc123')
  })

  it('sets combined flag', () => {
    const params = buildDryingRateParams('plant1', '', defaultCalibParams, 48, true)
    expect(params.get('combined')).toBe('true')
  })

  it('includes estimator only when not exp_mcmc', () => {
    const paramsExp = buildDryingRateParams('plant1', '', { ...defaultCalibParams, estimator: 'exponential' }, 48)
    expect(paramsExp.get('estimator')).toBe('exponential')

    const paramsMcmc = buildDryingRateParams('plant1', '', defaultCalibParams, 48)
    expect(paramsMcmc.has('estimator')).toBe(false)
  })
})

describe('buildWaterCalibrationParams', () => {
  it('includes mcmc params for exp_mcmc estimator', () => {
    const params = buildWaterCalibrationParams('plant1', '', defaultCalibParams)
    expect(params.get('n_burn')).toBe('10')
    expect(params.get('xmin_mu')).toBe('850')
  })

  it('includes prior_min/max for linear prior', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams, prior: 'linear' })
    expect(params.get('prior')).toBe('linear')
    expect(params.get('prior_min')).toBe('867')
  })

  it('includes prior_min/max for exp_mcmc even when prior is calibrated', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams, prior: 'calibrated' })
    expect(params.get('prior_min')).toBe('867')
  })

  it('excludes empty optional params', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams, nBurn: '', nSteps: '' })
    expect(params.has('n_burn')).toBe(false)
  })
})
