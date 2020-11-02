const imgSize = 500
const confThres = 0.1
let lastPose

// env setting
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

async function loadModel() {
  model = await posenet.load({
    architecture: 'MobileNetV1',
    outputStrides: 16,
    inputResolution: { width: imgSize, height: imgSize },
    muluplier: 0.75
  })
}

function detect(video) {
  const canvas = document.getElementById('output')
  const ctx = canvas.getContext('2d')
  console.log(ctx)
  canvas.width = imgSize
  canvas.height = imgSize

  async function getPose() {
    const pose = await model.estimateSinglePose(video, {
      flipHorizontal: true
    })
    lastPose = pose

    ctx.clearRect(0, 0, imgSize, imgSize)
    ctx.save()

    ctx.scale(-1, 1)
    ctx.translate(-imgSize, 0)
    ctx.drawImage(video, 0, 0, imgSize, imgSize)
    ctx.restore()

    drawKeypoints(pose.keypoints, confThres, ctx)
    drawSkeleton(pose.keypoints, confThres, ctx)
    requestAnimationFrame(getPose)
  }

  getPose()
}

// drawing part
function drawPoint(ctx, y, x, r) {
  ctx.beginPath() // initilizer
  ctx.arc(x, y, r, 0, 2 * Math.PI)
  ctx.fillStyle = 'red'
  ctx.fill()
}

function drawSegment([ay, ax], [by, bx], scale, ctx) {
  ctx.beginPath()
  ctx.moveTo(ax * scale, ay * scale)
  ctx.lineTo(bx * scale, by * scale)
  ctx.lineWidth = '10px'
  ctx.strokeStyle = 'red'
  ctx.stroke()
}

function toTuple({ y, x }) {
  return [y, x]
}

function drawSkeleton(keypoints, confThres, ctx, scale = 1) {
  const adjKeypoints = posenet.getAdjacentKeyPoints(keypoints, confThres)
  adjKeypoints.forEach((keypoints) => {
    drawSegment(toTuple(keypoints[0].position), toTuple(keypoints[1].position),
      scale, ctx)
  })
}

function drawKeypoints(keypoints, confThres, ctx, scale = 1) {
  for (let i = 0; i < keypoints.length; i += 1) {
    const keypoint = keypoints[i]
    if (keypoint.score > confThres) {
      const { y, x } = keypoint.position
      drawPoint(ctx, y * scale, x * scale, 3)
    }
  }
}

async function init() {
  const video = await loadVideo()
  await loadModel()
  detect(video)
}

init()