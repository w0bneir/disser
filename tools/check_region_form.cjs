// Offline form-contract test. No browser, network, audio playback or downloads.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(process.argv[2]);
const html = fs.readFileSync(path.join(root, 'experiment/region_gate.html'), 'utf8');
const publicData = JSON.parse(fs.readFileSync(path.join(root, 'experiment/manifest_public.json'), 'utf8'));
const code = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const storage = new Map();
let lastBlob;
let downloadCount = 0;
function fakeSession() {
  const fields = ['difference', 'identity', 'useful', 'artifacts', 'sequence', 'comment'];
  const cards = Object.keys(publicData.comparisons).map(id => {
    const controls = fields.map(field => ({dataset: {field}, value: '', handlers: {},
      addEventListener(event, fn) {this.handlers[event] = fn;}, focus() {this.focused = true;}}));
    return {dataset: {id}, controls, querySelectorAll() {return controls;}};
  });
  const controls = cards.flatMap(card => card.controls);
  const audio = [0, 1].map(() => ({paused: true, currentTime: 0, handlers: {},
    addEventListener(event, fn) {this.handlers[event] = fn;}, pause() {this.paused = true;}}));
  const status = {textContent: ''};
  const save = {addEventListener(event, fn) {this[event] = fn;}};
  const document = {
    getElementById(id) {return id === 'experiment-data' ? {textContent: JSON.stringify(publicData)} : id === 'status' ? status : save;},
    querySelectorAll(selector) {
      if (selector === 'section[data-id]') return cards;
      if (selector === '[data-field]') return controls;
      if (selector === 'select[data-field]') return controls.filter(c => c.dataset.field !== 'comment');
      if (selector === 'audio') return audio;
      throw new Error('Unrecognized selector ' + selector);
    },
    querySelector(selector) {return cards.find(c => selector.includes('"' + c.dataset.id + '"'));},
    createElement() {return {click() {downloadCount++;}};},
  };
  const sandbox = {document, crypto: {randomUUID: () => 'AUTOMATED-TEST-NOT-A-LISTENER'},
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    Blob: class {constructor(parts) {lastBlob = parts.join('');}},
    URL: {createObjectURL: () => 'mock-download', revokeObjectURL() {}},
    setTimeout: fn => fn()};
  vm.runInNewContext(code, sandbox);
  return {cards, controls, audio, status, save};
}
const first = fakeSession();
first.save.click();
assert.equal(downloadCount, 0);
assert.ok(first.status.textContent.includes('Заполните'));
for (const control of first.controls) {
  control.value = {difference: 'none', identity: 'yes', useful: 'none', artifacts: 'none', sequence: 'tie', comment: 'AUTOMATED QA'}[control.dataset.field];
  control.handlers.change();
}
const restored = fakeSession();
assert.ok(restored.controls.every(c => c.value !== ''));
restored.audio[0].paused = false;
restored.audio[0].currentTime = 1;
restored.audio[1].handlers.play();
assert.equal(restored.audio[0].paused, true);
assert.equal(restored.audio[0].currentTime, 0);
restored.save.click();
assert.equal(downloadCount, 1);
const output = JSON.parse(lastBlob);
assert.equal(output.package_id, publicData.package_id);
assert.deepEqual(output.stimulus_sha256, publicData.audio_sha256);
assert.equal(output.ratings.length, 5);
for (const row of output.ratings) assert.deepEqual(row.assets, publicData.comparisons[row.id]);
console.log('PASS: incomplete-form guard, saved answers restoration, single-player handler, package-bound JSON export. Offline mock only; actual browser playback not tested.');
