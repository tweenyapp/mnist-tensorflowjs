var trainPath = 'http://mnistdemo.tweeny.in/mnist_data/'
var testPath = 'http://mnistdemo.tweeny.in/mnist_data/mnist_batch_20.png'

const model = tf.sequential();

model.add(tf.layers.conv2d({
	inputShape: [28, 28, 1],
	kernelSize: 5,
	filters: 8,
	strides: 1,
	activation: 'relu',
	kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
model.add(tf.layers.conv2d({
	kernelSize: 5,
	filters: 16,
	strides: 1,
	activation: 'relu',
	kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense(
	{units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

const LEARNING_RATE = 0.01;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
	optimizer: optimizer,
	loss: 'categoricalCrossentropy',
	metrics: ['accuracy'],
});

const BATCH_SIZE = 30;
const NUM_EPOCHS = 1;

async function showPredictions(testArray, testImage) {
	var labels = [], 
		inps = [];
		images = [];
	for (var i=0; i<20; i++){
		randomIndex = Math.floor(Math.random() * 3000);
		labels.push(testLabels[randomIndex]);
		inps.push(...testArray.slice(randomIndex*784, (randomIndex+1)*784));
		images.push(...testImage.data.slice(randomIndex*784*4, (randomIndex+1)*784*4));
	}

	const xs = tf.tensor2d(inps, [20, IMAGE_SIZE]);
	const output = model.predict(xs.reshape([-1, 28, 28, 1]));
	const axis = 1;
	const predictions = Array.from(output.argMax(axis).dataSync());
	await show(images, labels, predictions)
}

function plotGraph(losses) {
	var canvas = document.getElementById('trainGraph');
	ctx = canvas.getContext('2d');
	var H = canvas.height;
	var W = canvas.width;
	ctx.clearRect(0, 0, W, H);
	var ng = 10;
	var pad = 25;	
	
	var step = 3000;
	if (losses.length > 10) var maxx = losses.length;
	else var maxx = 10;
	var maxy = Math.max(...losses)

	//Draw a grid on the canvas
	ctx.strokeStyle =  "#999";
	for(var i=0; i<ng+1; i++){
		var xPos = pad+(W-2*pad)*i/ng
		ctx.moveTo(xPos, pad)
		ctx.lineTo(xPos, H-pad)
		var text = (i/ng)*maxx
		ctx.fillText((text*step/1000).toFixed(1)+'k', xPos-10, H-pad+14)
	}
	for(var i=0; i<ng+1; i++){
		var yPos = pad+(H-2*pad)*i/ng
		ctx.moveTo(pad, yPos)
		ctx.lineTo(W-pad, yPos)
		var text = ((ng-i)*maxy)/ng
		ctx.fillText(text.toFixed(2), 0, yPos)
	}
	ctx.stroke();

	//Plot the actual loss function
	
	ctx.strokeStyle = "red";
	ctx.beginPath();
	for (var i=0; i<losses.length; i++){
		var yPos = H - ((losses[i]/maxy)*(H-2*pad) + pad);
		var xPos = ((i+1)/maxx)*(W-2*pad) + pad;
		if (i === 0) ctx.moveTo(xPos,yPos);
		else ctx.lineTo(xPos,yPos);
	}
	ctx.stroke();
}


async function train(){
	var testImage = await load(testPath);
	var testArray = new Float32Array(testImage.width*testImage.height)
	for (var j=0; j<testImage.data.length; j++) {
		testArray[j] = testImage.data[j * 4] / 255;
	}
	var losses = [];
	var exampleSeen = 0;
	for (var epochno=0; epochno<NUM_EPOCHS; epochno++){
		for (var i=0; i<20; i++) {
			var imageData = await load(trainPath + 'mnist_batch_' + i +'.png');
			var datasetBytesView = new Float32Array(imageData.width*imageData.height)
			
			for (var j=0; j<imageData.data.length; j++) {
				datasetBytesView[j] = imageData.data[j * 4] / 255;
			}

			var lossVal = 0,
				count = 0
			for (var imageIndex=0; imageIndex<imageData.height; imageIndex+=BATCH_SIZE) {
				var labelIndex = imageIndex+(3000*i)
				var batch = await get_batch(BATCH_SIZE, datasetBytesView, trainLabels, imageIndex, labelIndex);
				const history = await model.fit(
				batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
				{batchSize: BATCH_SIZE, epochs: 1});
				const loss = history.history.loss[0];
				lossVal+=loss;
				count+=1
			}
			exampleSeen+=3000;
			document.getElementById('example').innerHTML = exampleSeen
			losses.push(lossVal/count)
			document.getElementById('loss').innerHTML = lossVal/count
			await plotGraph(losses)
			await showPredictions(testArray, testImage)
		}
	}
}

//var losses = [2.215643181800842,1.70023361086845,0.871379427015781,0.493703858852386,0.49132644504308,0.394590208232402,0.3195764508843421,0.2905453762784,0.320307144075632,0.28078611835837364,0.30876618791371585,0.22582871098071336,0.2367681137099862,0.23873020108789206,0.22747866369783878,0.23279226860031485,0.2199552042968571,0.19789801463484763,0.1539139676094055,0.13208426334895193]
//plotGraph(losses);
train();