import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SensorsPane from './SensorsPane'

const defaultCalibrator = {
  params: {
    offsetMin: '5', widthMin: '50', estimator: 'exp_mcmc', priorWeight: '1.0', nBurn: '10', nSteps: '30',
    systemCapacityMean: '', systemCapacityStd: '', emaTauMin: '60',
  },
  setParam: vi.fn(),
  calibration: null,
  calibLoading: false,
  swcLoading: false,
  calibError: null,
  mappedSeries: [],
  mappedBands: [],
  dryingRate: null,
  dryingRateLoading: false,
  calculate: vi.fn(),
}

const defaultPlantSensors = [
  { plant_name: 'plant1', device_address: 'dev1', sensor: 'cap1' },
  { plant_name: 'plant1', device_address: 'dev2', sensor: 'cap2' },
]

const defaultProps = {
  plantFilter: 'plant1',
  sensorFilter: 'dev1:cap1',
  calibrator: defaultCalibrator,
  plantSensors: defaultPlantSensors,
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

  it('shows calibError in water mode when no mappedSeries', () => {
    renderWithProps({
      calibrator: { ...defaultCalibrator, calibError: 'Calibration failed' },
    })
    fireEvent.click(screen.getByText('Water (ml)'))
    expect(screen.getByText(/Calibration failed/)).toBeTruthy()
  })

  it('shows water chart when calibError set but mappedSeries has data', () => {
    renderWithProps({
      calibrator: {
        ...defaultCalibrator,
        calibError: 'Calibration failed',
        mappedSeries: [{ label: 'water', points: [{ t: 1000, v: 10, raw: 10 }], color: '#000' }],
        mappedBands: [],
      },
    })
    fireEvent.click(screen.getByText('Water (ml)'))
    expect(screen.getByText(/Calibration warning/)).toBeTruthy()
    expect(screen.queryByText(/Click Recalculate/)).toBeNull()
  })

  it('does NOT block mV mode when calibError is set', () => {
    renderWithProps({
      calibrator: { ...defaultCalibrator, calibError: 'Calibration failed' },
    })
    expect(screen.queryByText(/Calibration failed/)).toBeNull()
    expect(screen.getByText(/No data in range/)).toBeTruthy()
  })

  it('shows loading when swcLoading is true even if calibLoading is false', () => {
    renderWithProps({
      calibrator: { ...defaultCalibrator, swcLoading: true, calibLoading: false },
    })
    fireEvent.click(screen.getByText('Water (ml)'))
    expect(screen.getByText(/Computing calibration/)).toBeTruthy()
  })
})

describe('SensorsPane auto-recalculate', () => {
  beforeEach(() => vi.clearAllMocks())

  it('calls calculate on mount when plantFilter is set', () => {
    renderWithProps()
    expect(defaultCalibrator.calculate).toHaveBeenCalledTimes(1)
    expect(defaultCalibrator.calculate).toHaveBeenCalledWith(
      expect.objectContaining({
        plantFilter: 'plant1',
        sensorFilter: 'dev1:cap1',
        rangeHours: 48,
        plantSensors: defaultPlantSensors,
      })
    )
  })

  it('does not call calculate on mount when plantFilter is empty', () => {
    renderWithProps({ plantFilter: '' })
    expect(defaultCalibrator.calculate).not.toHaveBeenCalled()
  })

  it('passes plantSensors to calculate', () => {
    renderWithProps()
    const callArgs = defaultCalibrator.calculate.mock.calls[0][0]
    expect(callArgs.plantSensors).toBe(defaultPlantSensors)
  })
})