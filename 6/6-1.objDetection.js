let ctx
const imgSize = 500
let scoreThreshold = 0.9
let iouThreshold = 0.5
let topkThreshold = 10

// set up camera default codes from 4
function setupCamera() {
  const video = document.getElementById('video')
  video.width = imgSize
  video.height = imgSize

  navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: imgSize,
      height: imgSize,
    }
  }).then((stream) => {
    video.srcObject = stream
  })

  return new Promise((resolve) => {
    video.onloadeddata = () => resolve(video)
  })
}

async function loadVideo() {
  const video = await setupCamera()
  video.play()

  return video
}

// modele detect setting
function detect(model, video) {
  const canvas = document.getElementById('output')
  ctx = canvas.getContext('2d')

  canvas.width = imgSize
  canvas.height = imgSize

  async function getBoundingBoxes() {
    const preds = await model.detect(video, {
      score: scoreThreshold,
      iou: iouThreshold,
      topk: topkThreshold
    })

    ctx.save()
    ctx.scale(-1, 1)
    ctx.translate(-imgSize, 0)
    ctx.drawImage(video, 0, 0, imgSize, imgSize)
    ctx.restore()

    preds.forEach((pred) => {
      drawBoundingBoxes(pred)
    })

    requestAnimationFrame(getBoundingBoxes)
  }

  getBoundingBoxes()
}

function drawBoundingBoxes(pred) {
  ctx.font = '20px Arial'
  const {
    left,
    top,
    width,
    height
  } = pred.box

  ctx.strokeStyle = '#3498eb'
  ctx.lineWidth = 1
  ctx.strokeRect(left, top, width, height)

  // Draw the label background
  ctx.fillStyle = '#3498eb'
  const textWidth = ctx.measureText(prediction.label).width
  const textHeight = parseInt(ctx.font, 10)

  // Top left rectangle
  ctx.fillRect(left, top, textWidth + textHeight, textHeight * 2);
  // Bottom left rectangle
  ctx.fillRect(left, top + height - textHeight * 2, textWidth + textHeight, textHeight * 2)

  // Draw labels and score
  ctx.fillStyle = '#000000'
  ctx.fillText(prediction.label, left, top + textHeight)
  ctx.fillText(prediction.score.toFixed(2), left, top + height - textHeight)
}

function updateSliders(metric, updateAttribute) {
  const slider = document.getElementById(`${metric}-range`)
  const output = document.getElementById(`${metric}-value`)
  output.innerHTML = slider.value
  updateAttribute(slider.value)

  slider.oninput = function oninputCb() {
    output.innerHTML = this.value
    updateAttribute(this.value)
  }
}

async function init() {
  const model = await tf.automl.loadObjectDetection('model/model.json')
  const video = await loadVideo()
  console.log(model)
  detect(model, video)

  updateSliders('score', (value) => {
    iouThreshold = parseInt(value, 10)
  })

  updateSliders('iou', (value) => {
    iouThreshold = parseFloat(value);
  });

  updateSliders('topk', (value) => {
    topkThreshold = parseInt(value, 10);
  });
}

init()