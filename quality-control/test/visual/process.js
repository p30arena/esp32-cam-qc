const { Interpreter } = require("node-tflite");

const model = new Interpreter(require('fs').readFileSync("../efficientnet/out/model/model.tflite"));
model.allocateTensors();

console.log(model.inputs[0].type);
console.log(model.inputs[0].byteSize);
console.log(model.outputs[0].type);
console.log(model.outputs[0].byteSize);

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

process.on('message', (m) => {
  m = Buffer.from(m.data);
  const img = new Int8Array(m.length);
  for (let i = 0; i < m.length; i++) {
    img[i] = m[i] - 128;
  }
  model.inputs[0].copyFrom(img);
  model.invoke();
  const res = new Int8Array(1);
  model.outputs[0].copyTo(res);
  const prob = sigmoid(res);
  const lbl = ["error", "ok"][prob < 0.5 ? 0 : 1];
  process.send(lbl);
});