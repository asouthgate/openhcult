import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import SensorsPane from './SensorsPane'

const defaultProps = {
  plantFilter: 'plant1',
  sensorFilter: 'abc123:temperature',
  sensorAssignedAt: null,
  calibration: null,
  calibParams: {
    offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
    estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
    systemCapacityMean: '', systemCapacityStd: '', emaTauMin: '60',
  },
  setCalibParam: vi.fn(),
  plantSensors: [],
  combinedSwc: null,
  calibLoading: false,
  calibError: null,
  dryingRate: null,
  rangeHours: 48,
  setRangeHours: vi.fn(),
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
    renderWithProps({ calibError: 'Invalid system_capacity_mean value' })
    expect(screen.getByText('Invalid system_capacity_mean value')).toBeTruthy()
  })

  it('shows error even when stale calibration data exists', () => {
    renderWithProps({
      calibration: { prior_x: [400, 500], mean: [10, 20], scale: 1 },
      calibError: 'Some error occurred',
    })
    expect(screen.getByText('Some error occurred')).toBeTruthy()
  })

  it('shows loading state when calibLoading is true', () => {
    renderWithProps({ calibLoading: true, calibError: null })
    expect(screen.getByText(/Loading/)).toBeTruthy()
  })
})
