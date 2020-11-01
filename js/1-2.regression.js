// load required 
const csvUrl = 'https://gist.githubusercontent.com/juandes/2f1ffa32dd4e58f9f5825eca1806244b/raw/c5b387382b162418f051fd83d89fddb4067b91e1/steps_distance_df.csv';
let dataset, model, fittedLinePlot

const dataSurface = {
  name: 'sactter plot',
  tabel: "Data"
}

const fittedSurface = {
  name: 'fitted', // titles
  tab: 'fit' // tab name
}

const dataToVis = []
const predToVis = []

// load pre-stored data 
// TODO: datalaoder
async function loadData() {
  dataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        distance: { // col-name of target
          isLabel: true,
        },
      },
    },
  );

  // optional for catch data structure
  await dataset.forEachAsync((e) => {
    dataToVis.push({
      x: e.xs.steps,
      y: e.ys.distance
    })
  });

  tfvis.render.scatterplot(dataSurface, {
    values: [dataToVis],
    series: ['Dataset']
  })
}

// create buttom to do something intended
function createLoadButton() {
  const btn = document.createElement('button')
  btn.innerText = 'load data from pre-defined url'

  btn.addEventListener('click', () => {
    loadData()
    const trainBtn = document.getElementById('train-btn')
    trainBtn.disabled = false
  })

  document.querySelector('#visualize').appendChild(btn);
}

function createTrainButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Train!'
  btn.disabled = true
  btn.id = 'train-btn'

  btn.addEventListener('click', () => {
    const numberEpochs = document.getElementById('epochs').value;
    console.log(numberEpochs);
    Train(parseInt(numberEpochs, 10));
  });

  document.querySelector('#train').appendChild(btn);
}

// build tensorflow model
async function buildModel() {
  const numOfFeatures = (await dataset.columnNames()).length - 1;
  // Define the model.
  model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1,
    activation: 'linear',
  }));

  model.compile({
    optimizer: tf.train.adam(1e-2),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  model.summary();

  return model
}

async function Train(epochs) {
  writer('start training')
  const flattenedDataset = dataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    // Convert the features (xs) and labels (ys) to an array
    .batch(10)
    .shuffle(100); // buffer size and seed

  model = await buildModel()
  // Fit the model
  const history = await
    model.fitDataset(flattenedDataset, {
      epochs: epochs,
      callbacks: {
          onEpochEnd: async (epoch, logs) => {
              console.log(`iter: ${epoch}, loss:${logs.loss}`)
              writer(`iter: ${epoch}, loss:${logs.loss}`)
          }
        }
      });

  console.log('finish train')
  plotFitted(0, 30000, 500)
}

// optional fitted value plot
function plotFitted(min, max, steps) {
  tfvis.visor().open();
  const fittedLinePoints = [];
  const predictors = Array.from(
    {length: (max-min) / steps + 1},
    (_, i) => min + (i*steps)
  )

  const preds = model.predict(tf.tensor1d(predictors)).dataSync()

  predictors.forEach((value, i) => {
    fittedLinePoints.push({
      x: value, y: preds[i]
    })
  })

  const structureToVis = {
    values: [dataToVis, fittedLinePoints],
    tab: ['true', 'preds']
  }

  tfvis.render.scatterplot(fittedSurface, structureToVis)
}

// utils for logging on html(p-tag)
function writer(x) {
  const text = document.createElement('p')
  text.innerHTML = x
  document.querySelector('#train').appendChild(text);
}

// like if name == main
function init() {
  createTrainButton();
  createLoadButton();
}

init()

// //   build custom activation
// class Mish extends tf.layers.activation {
//     static get className() {
//         return 'mish'
//     }

//     apply(x) {
//         return tf.tidy(() => {
//             tf.math.tanh(tf.math.softplus(x))
//         })
//     }
// }
// tf.serialization.registerClass(Mish); 