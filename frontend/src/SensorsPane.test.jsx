import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import SensorsPane from './SensorsPane'

const defaultProps = {
  plantFilter: 'plant1',
  sensorFilter: 'abc123:temperature',
  sensorAssignedAt: null,
  calibrationMl: null,
  calibrationFrac: null,
  calibParams: {
    offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
    estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
    systemCapacityMean: '', systemCapacityStd: '', emaTauMin: '60',
  },
  setCalibParam: vi.fn(),
  plantSensors: [],
  combinedSwcMl: null,
  combinedSwcFrac: null,
  calibLoadingMl: false,
  calibLoadingFrac: false,
  calibErrorMl: null,
  calibErrorFrac: null,
  dryingRate: null,
  rangeHours: 48,
  setRangeHours: vi.fn(),
  recalculateMl: vi.fn(),
  recalculateFrac: vi.fn(),
}

vi.mock('./api', () => ({
  apiJson: vi.fn(() => Promise.resolve({ data: [] })),
  apiFetch: vi.fn(() => Promise.resolve({ ok: true })),
}))

function renderWithProps(overrides = {}) {
  return render(<SensorsPane {...defaultProps} {...overrides} />)
}

describe('SensorsPane error handling', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows calibError regardless of measure mode', () => {
    renderWithProps({ calibErrorMl: 'Invalid system_capacity_mean value' })
    expect(screen.getByText('Invalid system_capacity_mean value')).toBeTruthy()
  })

  it('shows error even when stale calibration data exists', () => {
    renderWithProps({
      calibrationMl: { prior_x: [400, 500], mean: [10, 20], scale: 1 },
      calibErrorMl: 'Some error occurred',
    })
    expect(screen.getByText('Some error occurred')).toBeTruthy()
  })

  it('shows loading state when calibLoading is true', () => {
    renderWithProps({ calibLoadingMl: true, calibErrorMl: null })
    expect(screen.getByText(/Loading/)).toBeTruthy()
  })
})
