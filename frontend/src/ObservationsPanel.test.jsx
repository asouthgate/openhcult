import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ObservationsPanel from './ObservationsPanel'

vi.mock('./api', () => ({
  apiJson: vi.fn(),
  apiFetch: vi.fn(),
}))

const defaultProps = {
  observations: [],
  calibration: null,
  setObservations: vi.fn(),
  pendingTime: null,
  pendingPlant: '',
  pendingMl: '',
  onMlChange: vi.fn(),
  onCancel: vi.fn(),
}

function renderWithProps(overrides = {}) {
  return render(<ObservationsPanel {...defaultProps} {...overrides} />)
}

describe('ObservationsPanel watering event form', () => {
  it('shows nothing when pendingTime is null', () => {
    renderWithProps()
    expect(screen.queryByText(/Select a plant/)).toBeNull()
    expect(screen.queryByText(/Watering at/)).toBeNull()
  })

  it('shows select-plant message when pendingTime is set but no plant selected', () => {
    renderWithProps({ pendingTime: Date.now(), pendingPlant: '' })
    expect(screen.getByText(/Select a plant to record a watering event/)).toBeTruthy()
    expect(screen.queryByText(/Watering at/)).toBeNull()
    expect(screen.queryByPlaceholderText('ml')).toBeNull()
  })

  it('shows full form when pendingTime and pendingPlant are both set', () => {
    renderWithProps({ pendingTime: Date.now(), pendingPlant: 'Ficus' })
    expect(screen.getByText(/Watering at/)).toBeTruthy()
    expect(screen.getByPlaceholderText('ml')).toBeTruthy()
    expect(screen.getByText('Record')).toBeTruthy()
    expect(screen.queryByText(/Select a plant/)).toBeNull()
  })

  it('disables Record button when ml is empty', () => {
    renderWithProps({ pendingTime: Date.now(), pendingPlant: 'Ficus', pendingMl: '' })
    expect(screen.getByText('Record').disabled).toBe(true)
  })

  it('enables Record button when ml is entered', () => {
    renderWithProps({ pendingTime: Date.now(), pendingPlant: 'Ficus', pendingMl: '100' })
    expect(screen.getByText('Record').disabled).toBe(false)
  })

  it('calls onCancel when Cancel is clicked in full form', () => {
    const onCancel = vi.fn()
    renderWithProps({ pendingTime: Date.now(), pendingPlant: 'Ficus', onCancel })
    fireEvent.click(screen.getByText('Cancel'))
    expect(onCancel).toHaveBeenCalledOnce()
  })

  it('calls onMlChange when ml input changes', () => {
    const onMlChange = vi.fn()
    renderWithProps({ pendingTime: Date.now(), pendingPlant: 'Ficus', onMlChange })
    fireEvent.change(screen.getByPlaceholderText('ml'), { target: { value: '50' } })
    expect(onMlChange).toHaveBeenCalled()
  })
})
