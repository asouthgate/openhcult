import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import SensorsPane from './SensorsPane'

const defaultCalibrator = {
  params: {
    offsetMin: '5', widthMin: '50', prior: 'calibrated', priorMin: '867', priorMax: '2009',
    estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
    systemCapacityMean: '', systemCapacityStd: '', emaTauMin: '60',
  },
  setParam: vi.fn(),
  calibration: null,
  calibLoading: false,
  calibError: null,
  combinedSwc: null,
  dryingRate: null,
  dryingRateLoading: false,
  calculate: vi.fn(),
}

const defaultProps = {
  plantFilter: 'plant1',
  sensorFilter: 'abc123:temperature',
  calibrator: defaultCalibrator,
}

vi.mock('./api', () => ({
  apiJson: vi.fn(() => Promise.resolve({ data: [] })),
  apiFetch: vi.fn(() => Promise.resolve({ ok: true })),
}))

vi.mock('./useSensorData', () => ({
  useSensorData: () => ({ series: [], observations: [], loading: false }),
}))

function renderWithProps(overrides = {}) {
  return render(<SensorsPane {...defaultProps} {...overrides} />)
}

describe('SensorsPane error handling', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows calibError regardless of measure mode', () => {
    renderWithProps({ calibrator: { ...defaultCalibrator, calibError: 'Invalid system_capacity_mean value' } })
    expect(screen.getByText('Invalid system_capacity_mean value')).toBeTruthy()
  })

  it('shows error even when stale calibration data exists', () => {
    renderWithProps({
      calibrator: { ...defaultCalibrator, calibration: { prior_x: [400, 500], mean: [10, 20], scale: 1 }, calibError: 'Some error occurred' },
    })
    expect(screen.getByText('Some error occurred')).toBeTruthy()
  })

  it('shows loading state when calibLoading is true', () => {
    renderWithProps({ calibrator: { ...defaultCalibrator, calibLoading: true, calibError: null } })
    expect(screen.getByText(/Computing calibration/)).toBeTruthy()
  })
})
