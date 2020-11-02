import { MnistData } from './3-1.data.js'

// define data and model property
let model, data
let isTrained = false
const imgSize = 28
const imgChannel = 1

// define plotting property
let ctx
const canvasSize = 400
let lastPosition = { x: 0, y: 0 }
let drawing = false

const dataSurface = { name: 'Sample', tab: 'Data' };

function buildModel() {
  model = tf.sequential()
  model.add(tf.layers.conv2d({
    inputShape: [imgSize, imgSize, imgChannel],
    kernelSize: 5,
    filters: 8,
    // strides: 1,
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2
  }))
  model.add(tf.layers.conv2d({
    // inputShape: [imgSize, imgSize, imgChannel], 기존 처럼 처음에만 붙어있으면 된다.
    kernelSize: 5,
    filters: 16,
    // strides: 1,
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2
  }))
  model.add(tf.layers.dropout({ rate: 0.2 }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
  }))

  model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'adam',
      metrics: ['acc']
    })
    // return model
}

async function train() {
  tfvis.visor().open();
  const testSize = 1000
  const trainSize = 5000
  const batchSize = 512

  // use tf.tidy for memory efficeince (no garbage collector options)
  const [trainX, trianY] = tf.tidy(() => {
    const d = data.nextTestBatch(trainSize)
    return [
      d.xs.reshape([trainSize, imgSize, imgSize, imgChannel]),
      d.labels
    ]
  })

  const [testX, testY] = tf.tidy(() => {
    const d = data.nextTestBatch(testSize)
    return [
      d.xs.reshape([testSize, imgSize, imgSize, imgChannel]),
      d.labels
    ]
  })

  await model.fit(trainX, trianY, {
    batchSize: batchSize,
    epochs: 5,
    validationData: [testX, testY],
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks({ name: 'loss & acc', tab: 'training' }, ['loss', 'val_loss', 'acc', 'val_acc'])
  })

  isTrained = true
}

function prepareCanvas() {
  const canvas = document.getElementById('draw-canvas')
  canvas.width = canvasSize
  canvas.height = canvasSize
    // Set the canvas style
  ctx = canvas.getContext('2d')
  ctx.strokeStyle = 'white'
  ctx.fillStyle = 'white'
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.lineWidth = 15
    // Add the canvas event listeners for mouse events => drawing handwriting
  canvas.addEventListener('mousedown', (e) => {
    drawing = true
    lastPosition = { x: e.offsetX, y: e.offsetY }
  })
  canvas.addEventListener('mouseout', () => {
    drawing = false
  })
  canvas.addEventListener('mousemove', (e) => {
    if (!drawing) { return }
    ctx.beginPath()
    ctx.moveTo(lastPosition.x, lastPosition.y)
    ctx.lineTo(e.offesetX, e.offsetY)
    ctx.stroke()
    lastPosition = { x: e.offsetX, y: e.offsetY }
  })
  canvas.addEventListener('mouseup', () => {
    drawing = false
    if (!isTrained) { return }

    // show result 
    const toPred = tf.browser.fromPixels(canvas)
      .resizeBilinear([imgSize, imgSize])
      .mean(2) // rgb를 흑백으로 만들기 위해서 
      .expandDims()
      .expandDims(3)
      .toFloat()
      .div(255.)

    // dataSync to get value
    const pred = model.predict(toPred).dataSync()

    const p = document.getElementById('predict-output')
    p.innerHTML = `pred is ${tf.argMax(pred).dataSync()}`
    console.log(pred)
  })

}

async function draw() {
  const surface = tfvis.visor().surface(dataSurface)
  const results = []
  const n = 26
  let digit

  const sample = data.nextTestBatch(n)

  for (let i = 0; i < n; i += 1) {
    digit = tf.tidy(() => sample.xs
      .slice([i, 0], [1, sample.xs.shape[1]])
      .reshape([imgSize, imgSize, imgChannel]))

    const visCanvas = document.createElement('canvas')
    visCanvas.width = imgSize
    visCanvas.height = imgSize
    visCanvas.style = 'margin: 5px'
    await tf.browser.toPixels(digit, visCanvas)
    surface.drawArea.appendChild(visCanvas)
  }
  digit.dispose()
}

function createBtn(innerText, selector, id, listener, disabled = false) {
  const btn = document.createElement('button')
  btn.innerText = innerText
  btn.id = id
  btn.disabled = disabled

  btn.addEventListener('click', listener)

  document.querySelector(selector).appendChild(btn)
}

function enableBtn(selector) {
  document.getElementById(selector).disabled = false
}

function writer(x, selector) {
  const text = document.createElement('p')
  text.innerHTML = x
  document.querySelector(selector).appendChild(text);
}

function init() {
  prepareCanvas()
  createBtn('load', '#pipeline', 'load-btn', async() => {
    data = new MnistData
    await data.load()
    draw()
    enableBtn('train-btn')
    console.log('load done')
  })
  createBtn('train', '#pipeline', 'train-btn', async() => {
    buildModel()
    await train()
    console.log('train done')
  }, true)
  createBtn('clear', '#pipeline', 'clear-btn', () => {
    ctx.clearRect(0, 0, canvasSize, canvasSize)
    lastPosition = { x: 0, y: 0 }
    console.log('clear canvas')
  })
}

init()