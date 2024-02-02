//$(".markdown-preview").appendChild("result")

$("#rt").textContent="Waiting..."


var input = document.getElementById('img_upl');
var preview = document.querySelector('.preview');
////perview fig/////
input.style.opacity = 0;
input.addEventListener('change', updateImageDisplay);function updateImageDisplay() {
	while(preview.firstChild) {
		preview.removeChild(preview.firstChild);
	}

	if(input.files.length === 0) {
		var para = document.createElement('p');
		para.textContent = '未選擇任何檔案';
    para.style="line-height: 20em;";
		preview.appendChild(para);
	} 
	else {
		var para = document.createElement('p');
		var image = document.createElement('img');
    image.style='max-width: 100%;max-height: 100%;'
		image.src = window.URL.createObjectURL(input.files[0]);
		preview.appendChild(image);
		preview.appendChild(para);
	}
}

////////canvas///////
var cvs = document.getElementById("myCanvas");
var ctx = cvs.getContext('2d');
cvs.height = window.innerHeight/6;
cvs.width = window.innerWidth*0.7;


var imgObj = new Image();
///////send fig////////
$("#send").click(function(e){

  var base64 = Base64Image(input.files[0]); 
  base64.then(function(value){
    console.log(value)
    let dataJSON={}
    dataJSON["image"]=value
    
    $("#rt").text('Waiting');
    $.ajax({
      type: "POST",
      url: "/SmileDetect_upload",
      data: JSON.stringify(dataJSON),
      dataType: "json",
      contentType: "application/json;charset=utf-8",
      
      success: (data) => {
        //alert(data.msg)
        console.log(data.msg);
        console.log(data.result);
        //var result = document.createElement('img');
        //result.src = 'data:image/png;base64,' + data.result
        //result.style='max-width: 100%;max-height: 100%;margin:5px;'
        
        imgObj.src = 'data:image/png;base64,' + data.result;
        imgObj.onload = function(){
          //ctx.drawImage(imgObj, 0, 0,cvs.width,cvs.height);
          scaleToFit(imgObj)
        }

        $("#rt").text(`結果${data.score}`)

  
      },

      
    });
    
  })
})
function Base64Image(file) {
  return new Promise((resolve,reject)=>{
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

let points=[]

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



})

$("#red").click(function(e){
  console.log("erase")
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, cvs.width, cvs.height);
  //ctx.drawImage(imgObj, 0, 0,cvs.width,cvs.height);
  scaleToFit(imgObj)
  ctx.fillStyle = "blue";

})


function scaleToFit(img){//縮放圖片填畫布
    // get the scale
    var scale = Math.min(cvs.width / img.width, cvs.height / img.height);
    // get the top left position of the image
    var x = (cvs.width / 2) - (img.width / 2) * scale;
    var y = (cvs.height / 2) - (img.height / 2) * scale;
    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
}
