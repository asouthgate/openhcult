import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
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
  mappedSeries: [],
  mappedBands: [],
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
  useSensorData: () => ({ series: [], observations: [], loading: false, error: null }),
}))

function renderWithProps(overrides = {}) {
  return render(<SensorsPane {...defaultProps} {...overrides} />)
}

describe('SensorsPane error handling', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows calibError in water mode', () => {
    renderWithProps({
      calibrator: { ...defaultCalibrator, calibError: 'Calibration failed' },
    })
    fireEvent.click(screen.getByText('Water (ml)'))
    expect(screen.getByText(/Calibration failed/)).toBeTruthy()
  })

  it('shows loading state when calibLoading is true', () => {
    renderWithProps({ calibrator: { ...defaultCalibrator, calibLoading: true, calibError: null } })
    fireEvent.click(screen.getByText('Water (ml)'))
    expect(screen.getByText(/Computing calibration/)).toBeTruthy()
  })

  it('shows recalculate prompt when no calibration', () => {
    renderWithProps({ calibrator: { ...defaultCalibrator } })
    fireEvent.click(screen.getByText('Water (ml)'))
    expect(screen.getByText(/Click Recalculate/)).toBeTruthy()
  })
})
