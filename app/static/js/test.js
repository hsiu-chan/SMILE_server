const aspectFit = (imageWidth, imageHeight, canvasWidth, canvasHeight) => {
    const imageRate = imageWidth / imageHeight
    const canvasRate = canvasWidth / canvasHeight
    let [dx, dy, dw, dh] = []
    if (imageRate >= canvasRate) {
      dw = canvasWidth
      dh = canvasWidth / imageRate
    } else {
      dh = canvasHeight
      dw = canvasHeight * imageRate
    }
    dx = (canvasWidth - dw) / 2
    dy = (canvasHeight - dh) / 2
    return [dx, dy, dw, dh]
  }
  

  const aspectFill = (imageWidth, imageHeight, canvasWidth, canvasHeight) => {
    const imageRate = imageWidth / imageHeight
    const canvasRate = canvasWidth / canvasHeight
    let [sx, sy, sw, sh] = []
    if (imageRate >= canvasRate) {
      sw = imageHeight * canvasRate
      sh = imageHeight
      sx = (imageWidth - sw) / 2
      sy = 0
    } else {
      sh = imageWidth / canvasRate
      sw = imageWidth
      sx = 0
      sy = (imageHeight - sh) / 2
    }
    return [sx, sy, sw, sh]
  }
  
var arr = Array(6).fill(1);
console.count(arr)
