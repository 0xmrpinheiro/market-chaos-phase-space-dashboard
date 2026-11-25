import { logisticMap, successivePairs, uniformNoise } from './chaos.js';

const byId = (id) => document.getElementById(id);
const r = byId('parameter');
const observations = byId('observations');
const SVG = 'http://www.w3.org/2000/svg';
const axes = { left: 58, top: 17, width: 390, height: 246 };
let seed = 42;

function svgNode(name, attributes, text) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function plot(svg, pairs, description) {
  const { left, top, width, height } = axes;
  const x = (value) => left + value * width;
  const y = (value) => top + (1 - value) * height;
  const nodes = [svgNode('desc', {}, description)];
  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    nodes.push(
      svgNode('line', { x1: x(tick), x2: x(tick), y1: y(0), y2: y(1), class: 'gridline' }),
      svgNode('line', { x1: x(0), x2: x(1), y1: y(tick), y2: y(tick), class: 'gridline' }),
      svgNode('text', { x: x(tick), y: y(0) + 21, 'text-anchor': 'middle' }, String(tick)),
      svgNode('text', { x: x(0) - 10, y: y(tick) + 4, 'text-anchor': 'end' }, String(tick)),
    );
  }
  nodes.push(
    svgNode('rect', { x: left, y: top, width, height, class: 'frame' }),
    svgNode('text', { x: x(0.5), y: y(0) + 44, 'text-anchor': 'middle' }, 'Value at step t'),
    svgNode('text', { x: 16, y: y(0.5), transform: `rotate(-90 16 ${y(0.5)})`, 'text-anchor': 'middle' }, 'Value at step t+1'),
  );
  for (const [previous, next] of pairs) {
    nodes.push(svgNode('circle', { cx: x(previous), cy: y(next), r: 2.6, class: 'point' }));
  }
  svg.replaceChildren(...nodes);
}

function render() {
  const parameter = Number(r.value);
  const count = Number(observations.value);
  byId('parameter-value').textContent = parameter.toFixed(2);
  byId('observation-value').textContent = String(count);
  byId('seed').textContent = String(seed);
  plot(byId('logistic-plot'), successivePairs(logisticMap(count, parameter)),
    `${count - 1} successive pairs from the logistic map with r = ${parameter.toFixed(2)}.`);
  plot(byId('noise-plot'), successivePairs(uniformNoise(count, seed)),
    `${count - 1} successive pairs from a seeded uniform-noise illustration.`);
}

r.addEventListener('input', render);
observations.addEventListener('input', render);
byId('cycle-example').addEventListener('click', () => { r.value = '3.20'; render(); });
byId('chaos-example').addEventListener('click', () => { r.value = '4'; render(); });
byId('new-noise').addEventListener('click', () => { seed = ((seed + 1) >>> 0) || 1; render(); });
render();