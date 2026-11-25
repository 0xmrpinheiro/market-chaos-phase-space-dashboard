import assert from 'node:assert/strict';
import test from 'node:test';
import { logisticMap, successivePairs, uniformNoise } from '../docs/chaos.js';

test('logistic points are successive values on the update curve', () => {
  const pairs = successivePairs(logisticMap(300, 4));
  assert.equal(pairs.length, 299);
  for (const [x, y] of pairs) {
    assert.ok(x >= 0 && x <= 1);
    assert.equal(y, 4 * x * (1 - x));
  }
});

test('the 3.20 preset converges to two alternating values', () => {
  const [a, b, c, d] = logisticMap(4, 3.2);
  assert.ok(Math.abs(a - b) > 0.1);
  assert.ok(Math.abs(a - c) < 1e-8);
  assert.ok(Math.abs(b - d) < 1e-8);
});

test('noise is replayable, changes by seed, and stays within the plot axes', () => {
  const samples = uniformNoise(300, 42);
  assert.deepEqual(samples, uniformNoise(300, 42));
  assert.notDeepEqual(samples, uniformNoise(300, 43));
  assert.ok(samples.every((value) => value >= 0 && value < 1));
  assert.deepEqual(successivePairs([0, 0.1, 0.8]), [[0, 0.1], [0.1, 0.8]]);
});