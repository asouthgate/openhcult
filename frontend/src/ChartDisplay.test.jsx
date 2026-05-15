import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChartDisplay from './ChartDisplay'

const defaultChartProps = {
  series: [],
  bands: [],
  observations: [],
  rangeMs: 48 * 3600 * 1000,
  onTimePick: null,
  pendingTime: null,
  yLabel: 'Voltage (mV)',
  eventWindowOffset: 300000,
  eventWindowWidth: 3000000,
  loading: false,
  sensorError: null,
  calibError: null,
  calibLoading: false,
  isWaterMode: false,
  hasCalibration: false,
  plantFilter: 'plant1',
}

function renderWithProps(overrides = {}) {
  return render(<ChartDisplay {...defaultChartProps} {...overrides} />)
}

describe('ChartDisplay', () => {
  describe('raw/voltage mode (isWaterMode=false)', () => {
    it('shows sensorError when set', () => {
      renderWithProps({ sensorError: 'Network error' })
      expect(screen.getByText(/Sensor data error/)).toBeTruthy()
    })

    it('does NOT show calibError in voltage mode', () => {
      renderWithProps({ calibError: 'Calibration failed' })
      expect(screen.queryByText(/Calibration failed/)).toBeNull()
      expect(screen.getByText(/No data in range/)).toBeTruthy()
    })

    it('shows loading when loading is true', () => {
      renderWithProps({ loading: true })
      expect(screen.getByText(/Loading/)).toBeTruthy()
    })

    it('shows no data when series empty and no error', () => {
      renderWithProps({})
      expect(screen.getByText(/No data in range/)).toBeTruthy()
    })
  })

  describe('water mode (isWaterMode=true)', () => {
    it('shows calibError when set', () => {
      renderWithProps({ isWaterMode: true, calibError: 'Invalid capacity' })
      expect(screen.getByText(/Calibration failed/)).toBeTruthy()
    })

    it('does NOT show sensorError in water mode', () => {
      renderWithProps({ isWaterMode: true, sensorError: 'Network error' })
      expect(screen.queryByText(/Sensor data error/)).toBeNull()
    })

    it('shows loading when calibLoading is true', () => {
      renderWithProps({ isWaterMode: true, calibLoading: true })
      expect(screen.getByText(/Computing calibration/)).toBeTruthy()
    })

    it('shows recalculate prompt when no calibration', () => {
      renderWithProps({ isWaterMode: true })
      expect(screen.getByText(/Click Recalculate/)).toBeTruthy()
    })

    it('shows no plant prompt when plantFilter empty', () => {
      renderWithProps({ isWaterMode: true, plantFilter: '' })
      expect(screen.getByText(/Select a plant/)).toBeTruthy()
    })

    it('shows no water data when hasCalibration but series empty', () => {
      renderWithProps({ isWaterMode: true, hasCalibration: true })
      expect(screen.getByText(/No water data in range/)).toBeTruthy()
    })
  })
})
