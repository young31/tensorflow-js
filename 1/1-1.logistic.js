const csvUrl = 'https://gist.githubusercontent.com/juandes/ba58ef99df9bd719f87f807e24f7ea1c/raw/59f57af034c52bd838c513563a3e547b3650e7ba/lr-dataset.csv';
let dataset
let model

function loadData() {
  dataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        label: { // label is col-name of target
          isLabel: true,
        },
      },
    },
  );
}

function createTrainButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Train!';

  btn.addEventListener('click', () => {
    const numberEpochs = document.getElementById('epochs').value;
    console.log(numberEpochs);
    Train(parseInt(numberEpochs, 10));
  });

  document.querySelector('#train').appendChild(btn);
}

async function buildModel() {
  const numOfFeatures = (await dataset.columnNames()).length - 1;
  // Define the model.
  model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1,
    activation: 'sigmoid',
  }));

  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  // Print the summary to console
  model.summary();

  return model
}

async function Train(numberEpochs) {
  // numOfFeatures is the number of column or features minus the label column
  console.log('start training')

  const flattenedDataset = dataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    // Convert the features (xs) and labels (ys) to an array
    .batch(10)
    .shuffle(100); // buffer size and seed

  model = await buildModel()
    // Fit the model
  await model.fitDataset(flattenedDataset, {
    epochs: numberEpochs,
    callbacks: {
      onEpochEnd: async(epoch, logs) => {
        console.log(`iter: ${epoch}, loss:${logs.loss}`)
        writer(`iter: ${epoch}, loss:${logs.loss}`)
      }
    }
  });

  console.log('finish train')
}

function writer(x) {
  const text = document.createElement('p')
  text.textContent = x
  document.querySelector('#train').appendChild(text);
}

// like if name == main
function init() {
  createTrainButton();
  // createVisButton();
  loadData();
}

init()