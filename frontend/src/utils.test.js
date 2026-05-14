import { describe, it, expect } from 'vitest'
import { interp } from './utils'

describe('interp', () => {
  it('returns 0 for empty arrays', () => {
    expect(interp(0.5, [], [])).toBe(0)
  })

  it('clamps to first y when x is below range', () => {
    expect(interp(0, [1, 2, 3], [10, 20, 30])).toBe(10)
  })

  it('clamps to last y when x is above range', () => {
    expect(interp(5, [1, 2, 3], [10, 20, 30])).toBe(30)
  })

  it('interpolates midpoint', () => {
    expect(interp(2, [1, 3], [10, 30])).toBe(20)
  })
})
