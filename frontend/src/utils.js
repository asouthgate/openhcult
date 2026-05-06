export function sensorKey(deviceAddress, sensor) {
  return `${deviceAddress}:${sensor}`
}

export function sensorPart(key) {
  return key.slice(key.lastIndexOf(':') + 1)
}

export function parseSensorKey(key) {
  const sep = key.lastIndexOf(':')
  return { deviceAddress: key.slice(0, sep), sensor: key.slice(sep + 1) }
}

export function interp(x, xs, ys) {
  if (!xs || !ys || xs.length === 0) return 0
  if (x <= xs[0]) return ys[0]
  if (x >= xs[xs.length - 1]) return ys[xs.length - 1]
  let lo = 0, hi = xs.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid] <= x) lo = mid; else hi = mid
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo])
  return ys[lo] + t * (ys[hi] - ys[lo])
}
