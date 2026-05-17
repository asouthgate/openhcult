import { describe, it, expect } from 'vitest'
import { buildSwcTimeseriesParams, buildDryingRateParams, buildWaterCalibrationParams } from './paramsBuilder'

const defaultCalibParams = {
  offsetMin: '5', widthMin: '50', estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '100', nSteps: '200',
  systemCapacityMean: '', systemCapacityStd: '', emaTauMin: '60',
}

describe('buildSwcTimeseriesParams', () => {
  it('builds basic params', () => {
    const params = buildSwcTimeseriesParams('plant1', defaultCalibParams, 48)
    expect(params.get('plant')).toBe('plant1')
  })

  it('excludes empty optional params', () => {
    const params = buildSwcTimeseriesParams('plant1', { ...defaultCalibParams, systemCapacityMean: '', systemCapacityStd: '' }, 48)
    expect(params.has('system_capacity_mean')).toBe(false)
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
    expect(params.get('n_burn')).toBe('100')
    expect(params.get('n_steps')).toBe('200')
  })

  it('includes system_capacity params when provided', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams, systemCapacityMean: '1000', systemCapacityStd: '200' })
    expect(params.get('system_capacity_mean')).toBe('1000')
    expect(params.get('system_capacity_std')).toBe('200')
  })

  it('excludes system_capacity params when empty', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams, systemCapacityMean: '', systemCapacityStd: '' })
    expect(params.has('system_capacity_mean')).toBe(false)
    expect(params.has('system_capacity_std')).toBe(false)
  })

  it('includes return_fractional when true', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams }, true)
    expect(params.get('return_fractional')).toBe('true')
  })

  it('excludes return_fractional when false', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams }, false)
    expect(params.has('return_fractional')).toBe(false)
  })

  it('excludes empty optional params', () => {
    const params = buildWaterCalibrationParams('plant1', '', { ...defaultCalibParams, nBurn: '', nSteps: '' })
    expect(params.has('n_burn')).toBe(false)
  })

  describe('sensor filter handling', () => {
    it('buildWaterCalibrationParams omits sensor params when sensorFilter is empty', () => {
      const params = buildWaterCalibrationParams('plant1', '', defaultCalibParams)
      expect(params.has('sensor')).toBe(false)
      expect(params.has('device_address')).toBe(false)
    })

    it('buildWaterCalibrationParams parses sensor key when provided', () => {
      const params = buildWaterCalibrationParams('plant1', 'AA:BB:CC:cap1', defaultCalibParams)
      expect(params.get('sensor')).toBe('cap1')
      expect(params.get('device_address')).toBe('AA:BB:CC')
    })

    it('buildDryingRateParams omits sensor params when sensorFilter is empty', () => {
      const params = buildDryingRateParams('plant1', '', defaultCalibParams, 48)
      expect(params.has('sensor')).toBe(false)
      expect(params.has('device_address')).toBe(false)
    })

    it('buildDryingRateParams parses sensor key when provided', () => {
      const params = buildDryingRateParams('plant1', 'AA:BB:CC:cap1', defaultCalibParams, 48)
      expect(params.get('sensor')).toBe('cap1')
      expect(params.get('device_address')).toBe('AA:BB:CC')
    })
  })
})
