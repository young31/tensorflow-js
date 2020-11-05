// const dataURL = 'https://gist.githubusercontent.com/juandes/5c7397a2b8844fbbbb2434011e9d9cc5/raw/9a849143a3e3cb80dfeef3b1b42597cc5572f674/sequence.csv'
// const testURL = 'https://gist.githubusercontent.com/juandes/950003d00bd16657228e4cdd268a312a/raw/e5b5d052f95765d5bedfc6618e3c47c711d6816d/test.csv'

const dataURL = 'data/sequence.csv'
const testURL = 'data/test.csv'

const STEPS = 60
const dataSize = 900
const testSize = 8

let model

// Data
function loadData() {
  const trainSet = tf.data.csv(dataURL, {
    columnConfigs: {
      value: {
        isLabel: true,
      }
    }
  });

  const testSet = tf.data.csv(testURL, {
    columnConfigs: {
      value: {
        isLabel: true,
      }
    }
  });

  return { trainSet, testSet };
}

async function preprocessing(dataset, size) {
  // define data shape
  const sequences = tf.buffer([size, STEPS, 1])
  const targets = tf.buffer([size, 1])

  let row = 0;
  // make timeserires format
  await dataset.forEachAsync(({ xs, ys }) => {
    let col = 0
    Object.values(xs).forEach((element) => {
      sequences.set(element, row, col, 0)
      col += 1
    })

    targets.set(ys.value, row, 0)
    row += 1
  })

  return { xs: sequences.toTensor(), ys: targets.toTensor() }; // tensor 형태로 반환
}

// Model
async function buildModel() {
  model = tf.sequential()
  model.add(tf.layers.lstm({
    inputShape: [STEPS, 1],
    units: 32,
    returnSequences: false
  }))
  model.add(tf.layers.dense({ units: 1 }))
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.1)
  })
}


async function train(dataset) {
  const container = document.getElementById('canvas')
  await model.fit(dataset.xs, dataset.ys, {
    batchSize: 64,
    epochs: 10,
    validationSplit: 0.2,
    callbacks: [
      tfvis.show.fitCallbacks(container, ['loss', 'val_loss'], { callbakcs: ['onEpochEnd', 'onBatchEnd'] })
    ]
  })
}

async function predict(dataset) {
  return model
    .predict(dataset.xs)
    .dataSync()
}

function range(min, max, steps) {
  return Array.from({ length: (max - min) / steps + 1 }, (_, i) => min + i * steps)
}

// to HTML 
function createBtn(innerText, selector, id, listener, disabled = false) {
  const btn = document.createElement('button')

  btn.innerText = innerText
  btn.id = id
  btn.disabled = disabled

  btn.addEventListener('click', listener)

  document.querySelector(selector).appendChild(btn)
}


// plot
async function plotPrediction(which, testSet, predictions) {
  let testCase = (await testSet.xs.array());
  testCase = testCase[which].flat();

  // These are the Plotly traces to draw the test examples.
  // The first trace draws the test data.
  const traceSequence = {
    x: range(0, STEPS - 1, 1),
    y: testCase.slice(0, STEPS),
    mode: 'lines',
    type: 'scatter',
    name: 'Test data',
  };

  // The second trace draws the actual value.
  const traceActualValue = {
    x: [STEPS],
    y: [testCase[STEPS - 1]],
    mode: 'markers',
    type: 'scatter',
    name: 'Actual value',
    marker: {
      symbol: 'circle',
    },
  };

  // The second trace draws the predicted value.
  const tracePredictedValue = {
    x: [STEPS],
    y: [predictions[which]],
    mode: 'markers',
    type: 'scatter',
    name: 'Predicted value',
    marker: {
      symbol: 'diamond',
    },
  };

  const traces = [traceSequence, traceActualValue, tracePredictedValue];
  Plotly.newPlot('plot', traces);
}

async function init() {
  let predictions
  let { trainSet, testSet } = loadData()
  trainSet = await preprocessing(trainSet, dataSize)
  testSet = await preprocessing(testSet, testSize)

  console.log('data shape:', trainSet.xs.shape, testSet.xs.shape)

  const testIndex = range(1, 8, 1)

  testIndex.forEach((testCase) => {
    createBtn(`test case ${testCase}`, '#btn-test', `test-case-${testCase}`,
      async() => {
        plotPrediction(testCase - 1, testSet, predictions)
      }, true)
  })

  const trainBtn = document.getElementById('btn-train')

  trainBtn.addEventListener('click', async() => {
    await buildModel()
    await train(trainSet)
    predictions = await predict(testSet)

    testIndex.forEach((testCase) => {
      document.getElementById(`test-case-${testCase}`).disabled = false
    })
  })
}

init()