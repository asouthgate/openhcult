import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import DryingRateDisplay from './DryingRateDisplay'

vi.mock('./sensorDataTransforms', () => ({
  transformRateSeries: () => ({ mappedSeries: [], bands: [] }),
}))

const defaultProps = {
  dryingRate: null,
  dryingRateLoading: false,
  plantFilter: 'plant1',
  rangeHours: 48,
}

function renderWithProps(overrides = {}) {
  return render(<DryingRateDisplay {...defaultProps} {...overrides} />)
}

describe('DryingRateDisplay', () => {
  it('shows no plant prompt when plantFilter empty', () => {
    renderWithProps({ plantFilter: '' })
    expect(screen.getByText(/Select a plant/)).toBeTruthy()
  })

  it('shows loading when dryingRateLoading is true', () => {
    renderWithProps({ dryingRateLoading: true })
    expect(screen.getByText(/Computing drying rate/)).toBeTruthy()
  })

  it('shows error when dryingRate is null after loading', () => {
    renderWithProps({ dryingRate: null, dryingRateLoading: false })
    expect(screen.getByText(/Drying rate computation failed/)).toBeTruthy()
  })

  it('shows no data when dryingRate empty but not null', () => {
    renderWithProps({ dryingRate: { times_ms: [], rate_ml_per_day: [], valid: [], scale: 1 } })
    expect(screen.getByText(/No drying rate data available/)).toBeTruthy()
  })
})
