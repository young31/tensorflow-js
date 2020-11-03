const csvURL = 'https://gist.githubusercontent.com/juandes/34d4eb6dfd7217058d56d227eb077ca2/raw/c5c86ea7a32b5ae89ef06734262fa8eff25ee776/cluster_df.csv'
let dataset, model

const colMap = {
  0: 'black',
  1: 'green',
  2: 'blue',
  3: 'red',
};

const shapeMap = {
  0: 'circle',
  1: 'square',
  2: 'diamond',
  3: 'cross',
};

function loadData() {
  dataset = tf.data.csv(csvURL)
}

async function clustering(k) {
  const params = {
    k,
    maxIter: 200
  }

  model = ml5.kmeans(csvURL, params, plot)
}

function createClusteringButton() {
  const btn = document.createElement('button')
  btn.innerText = 'do cluster'

  btn.addEventListener('click', () => {
    const K = document.getElementById('k-range')
    clustering(K.value)
  })

  document.querySelector('#button').appendChild(btn)
}

function plot() {
  const x = []
  const y = []
  const colors = []
  const shapes = []

  model.dataset.forEach((e) => {
    x.push(e[0])
    y.push(e[1])
    colors.push(colMap[e.centroid])
    shapes.push(shapeMap[e.centroid])
  })

  const trace = {
    x,
    y,
    mode: 'markers',
    type: 'scatter',
    marker: { symbol: shapes, color: colors }
  }

  Plotly.newPlot('plot', [trace])
  writer('done', 'button')
}

function writer(x, target) {
  const text = document.createElement('p')
  text.innerHTML = x
  document.querySelector(`#${target}`).appendChild(text);
}

function init() {
  createClusteringButton()
}

init()