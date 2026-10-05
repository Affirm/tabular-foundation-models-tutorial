import assert from "node:assert/strict";
import { once } from "node:events";
import test from "node:test";
import { Worker } from "node:worker_threads";

const workerUrl = new URL("../materials/website/js/tabicl/worker.js", import.meta.url).href;

function browserWorker(t, setup = "") {
  const worker = new Worker(`
    const { parentPort } = require("node:worker_threads");
    globalThis.self = { postMessage: (message) => parentPort.postMessage(message) };
    ${setup}
    import(${JSON.stringify(workerUrl)}).then(() => {
      parentPort.on("message", (data) => self.onmessage({ data }));
    }).catch((error) => { throw error; });
  `, { eval: true });
  t.after(() => worker.terminate());
  return worker;
}

async function send(worker, message) {
  const response = once(worker, "message");
  worker.postMessage(message);
  const [result] = await response;
  return result;
}

test("worker failures retain the request type and tag", { timeout: 10_000 }, async (t) => {
  const worker = browserWorker(t);
  for (const type of ["prepare", "predict", "inspect", "grid"]) {
    const response = await send(worker, { type, tag: 123, X: [[0]], y: [0] });
    assert.equal(response.type, "error");
    assert.equal(response.requestType, type);
    assert.equal(response.tag, 123);
    assert.match(response.message, /Load|Prepare/);
  }
});

test("a failed artifact initialization cannot be reported as loaded on retry", { timeout: 10_000 }, async (t) => {
  const worker = browserWorker(t, `
    const manifest = {
      schema: "tabicl-browser-js/flat-tensors-v1", task: "classifier", config: {},
      binary_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      tensors: [{ name: "unknown_tensor", shape: [1] }],
    };
    globalThis.fetch = async (url) => String(url).endsWith("manifest.json")
      ? Response.json(manifest) : new Response(new Uint8Array());
  `);
  for (const tag of [1, 2]) {
    const response = await send(worker, { type: "load", tag });
    assert.equal(response.type, "error");
    assert.equal(response.requestType, "load");
    assert.equal(response.tag, tag);
    assert.match(response.message, /Unmapped upstream tensor key/);
  }
});

test("worker reports a failed manifest download before reading its body", { timeout: 10_000 }, async (t) => {
  const worker = browserWorker(t, `
    globalThis.fetch = async () => new Response("Not found", { status: 404 });
  `);
  const response = await send(worker, { type: "load", tag: 9 });
  assert.equal(response.type, "error");
  assert.equal(response.tag, 9);
  assert.match(response.message, /Manifest download failed \(404\)/);
});
