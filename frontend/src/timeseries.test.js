import { describe, it, expect } from 'vitest'
import { timeTicks, valueTicks } from './TimeseriesChart'

describe('timeTicks', () => {
  it('generates n+1 evenly spaced ticks', () => {
    expect(timeTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100])
  })
})

describe('valueTicks', () => {
  it('generates nice round ticks within range', () => {
    const ticks = valueTicks(0, 100, 5)
    expect(ticks).toContain(0)
    expect(ticks).toContain(100)
    expect(ticks.length).toBeGreaterThan(1)
  })
})
