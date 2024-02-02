//$(".markdown-preview").appendChild("result")

$("#rt").textContent = "Waiting..."




////////canvas///////
var cvs = document.getElementById("myCanvas");
var ctx = cvs.getContext('2d');
cvs.height = window.innerHeight / 6;
cvs.width = window.innerWidth * 0.7;
//fillStyle = "black";
//ctx.fillRect(0, 0, cvs.width, cvs.height);

function cvs_clear() {

  ctx.fillStyle = 'rgba(255, 255, 255, 0)';
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.fillStyle = "blue";

  //console.log(fig_r[0],fig_r[1],fig_r[2],fig_r[3])

  ctx.drawImage(document.getElementById('ini_fig'), fig_r[0], fig_r[1], fig_r[2], fig_r[3]);


}



function draw_pol(pol) {
  l = pol.length;
  ctx.beginPath();
  ctx.moveTo(pol[l - 1][0] * fig_r[4] + fig_r[0], pol[l - 1][1] * fig_r[4] + fig_r[1]);

  for (let i = 0; i < l; i += 1) {
    ctx.lineTo(pol[i][0] * fig_r[4] + fig_r[0], pol[i][1] * fig_r[4] + fig_r[1]);
  }
  ctx.fill();
}


function draw_mask(n) {
  cvs_clear();
  draw_pol(mask[n]);
  lb = document.getElementById('rt')
  console.log(lb.textContent);
  lb.textContent = `Mask${mask_now}，標籤${label[n]}`
}



//放圖片 
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

  r = dw / imageWidth;
  return [dx, dy, dw, dh, r]
}




///////get fig////////
let id
let mask;
let label;

let mask_now = 0;
var imgObj = new Image();
let fig_size;
let fig_r;
$("#getfig").click(function (e) {
  console.log("get figure")
  get_fig()

  /*$.ajax({
    type: "get",
    url: "/get_label_data",
    contentType: "application/json;charset=utf-8",

    success: (data) => {
      mask = data.mask;//設定mask
      id = data.id;

      mask_now = 0;
      label = Array(mask.length).fill(-1)
      fig_size = data.size;

      //放圖片
      let ini_fig = document.getElementById('ini_fig')
      ini_fig.src = 'data:image/png;base64,' + data.fig;
      ini_fig.style = 'width: 70%;max-height: 100%;margin:5px;'
      imgObj.src = 'data:image/png;base64,' + data.fig;


      //fig_r = cvs.width/imgObj.clientWidth;

      //console.log("fig",imgObj.clientWidth,imgObj.clientHeight,cvs.width,cvs.height)

      fig_r = aspectFit(fig_size[0], fig_size[1], cvs.width, cvs.height)

      //draw_mask(0)



    },
  });
  console.log(fig_size)*/



})


function get_fig() {
  $.ajax({
    type: "get",
    url: "/get_label_data",
    contentType: "application/json;charset=utf-8",

    success: (data) => {
      mask = data.mask;//設定mask
      id = data.id;

      mask_now = 0;
      label = Array(mask.length).fill(-1)
      fig_size = data.size;

      //放圖片
      let ini_fig = document.getElementById('ini_fig')
      ini_fig.src = 'data:image/png;base64,' + data.fig;
      ini_fig.style = 'width: 70%;max-height: 100%;margin:5px;'
      imgObj.src = 'data:image/png;base64,' + data.fig;


      //fig_r = cvs.width/imgObj.clientWidth;

      //console.log("fig",imgObj.clientWidth,imgObj.clientHeight,cvs.width,cvs.height)

      fig_r = aspectFit(fig_size[0], fig_size[1], cvs.width, cvs.height)

      //draw_mask(0)



    },
  });
  console.log(fig_size)
}

window.onload = function () {
  $("#getfig").click();
}

imgObj.onload = function () {
  draw_mask(mask_now)
}

function Base64Image(file) {
  return new Promise((resolve, reject) => {
    // 建立FileReader物件
    let reader = new FileReader()
    // 註冊onload事件，取得result則resolve (會是一個Base64字串)
    reader.onload = () => { resolve(reader.result) }
    // 註冊onerror事件，若發生error則reject
    reader.onerror = () => { reject(reader.error) }
    // 讀取檔案
    reader.readAsDataURL(file)
  })

}

/*let points=[]
$("canvas").click(function(e){
  let xPos = e.pageX - $(this).offset().left;
  let yPos = e.pageY - $(this).offset().top;
      
  ctx.fillStyle = "#0000ff";
  points.push([xPos,yPos]);
  //console.log([xPos,yPos]);
  ctx.fillRect(xPos-2, yPos-2, 4,4);

  let xy={};
  xy["xPos"]=xPos/cvs.width;
  xy["yPos"]=yPos/cvs.height;


  $("#show").html(`x: ${xPos}, y: ${yPos}<br>`);
  $.ajax({
    type: "GET",
    url: "/upload_img",
    data: (xy),
    dataType: "json",
    contentType: "application/json;charset=utf-8",
    
    success: (data) => {
      console.log(data.msg);
      let xpol=data.xpol.split(",");
      let ypol=data.ypol.split(",");
      let l=xpol.length;
      console.log(xpol,ypol);
      console.log(cvs.width,cvs.height)
      $("#rt").text(`結果${data.sc}`)

      ctx.beginPath();
      ctx.moveTo(xpol[l-1]*cvs.width,ypol[l-1]*cvs.height);
      for(let i =0;i<l;i+=1){
        ctx.lineTo(xpol[i]*cvs.width, ypol[i]*cvs.height);
      }
      ctx.fill();
    },

    
  });



})*/



$("#previous").click(function (e) {
  mask_now -= 1;
  if (mask_now < 0) { mask_now += mask.length; }
  console.log(mask_now);

  draw_mask(mask_now);

})

$("#next").click(function (e) {
  mask_now += 1;
  mask_now %= mask.length;
  console.log(mask_now);

  draw_mask(mask_now);

})


$("#check").click(function (e) {
  data = JSON.stringify({ "id": id, "mask": mask, "label": label })
  console.log(data)

  $.ajax({
    type: "post",
    url: "/get_label_data",
    contentType: "application/json",
    data: data,

    success: function (response) {
      console.log(response)
      get_fig()
    },

  });
})



//input
var input = document.getElementById("thisLabel");
var submitBtn = document.getElementById("submitBtn");

input.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    //draw_mask(0)
    event.preventDefault();
    document.getElementById("submitBtn").click();
    document.getElementById("next").click();
  }

});


function FsubmitBtn() {
  if (input.value.trim().length == 0) { return }
  label[mask_now] = input.value;
  draw_mask(mask_now)
  input.value = "";
}
submitBtn.addEventListener("click", FsubmitBtn);
