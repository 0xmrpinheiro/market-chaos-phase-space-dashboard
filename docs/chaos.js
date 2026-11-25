// Generators and plot inputs shared by the browser and Node tests.
export function logisticMap(count, r, start = 0.231) {
  let x = start;
  for (let i = 0; i < 100; i += 1) x = r * x * (1 - x);
  return Array.from({ length: count }, () => {
    const value = x;
    x = r * x * (1 - x);
    return value;
  });
}

export function uniformNoise(count, seed) {
  // Seeded xorshift32 for a replayable illustration, not a security RNG.
  let state = (seed >>> 0) || 1;
  return Array.from({ length: count }, () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  });
}

export function successivePairs(values) {
  return values.slice(0, -1).map((value, index) => [value, values[index + 1]]);
}