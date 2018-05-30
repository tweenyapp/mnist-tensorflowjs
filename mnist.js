var trainPath = 'http://localhost/images/'
var testPath = 'http://localhost/images/mnist_batch_20.png'

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

function showPredictions(testImage) {
	var labels = [], 
		images = [];
	for (var i; i<20; i++){
		randomIndex = Math.floor(Math.random() * 3000);
		labels.push(testLabels[randomIndex]);
		images.push(testImage.slice(randomIndex*784, (randomIndex+1)*784));
	}
	const xs = tf.tensor2d(images, [batchSize, IMAGE_SIZE]);
	//const labels = tf.oneHot(tf.tensor1d(mnistLabels.slice(labelIndex,labelIndex+batchSize), dtype='int32'), 10);
	const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));
	const axis = 1;
	const predictions = Array.from(output.argMax(axis).dataSync());

}

function plotGraph(losses) {
	var canvas = document.getElementById('trainGraph');
	var pad = 25;
	if (losses.length > 100) var maxx = losses.length;
	else var maxx = 100;
	var maxy = Math.max(...losses)
	ctx = canvas.getContext('2d');
	ctx.clearRect(0, 0, W, H);
	var H = canvas.height;
	var W = canvas.width;
	ctx.strokeStyle = "red";
	ctx.beginPath();
	for (var i=0; i<losses.length; i++){
		var yPos = H - ((losses[i]/maxy)*(H-2*pad) + pad);
		var xPos = (i/maxx)*(W-2*pad) + pad;
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
	for (var epochno=0; epochno<NUM_EPOCHS; epochno++){
		for (var i=0; i<20; i++) {
			var imageData = await load(trainPath + 'mnist_batch_' + i +'.png');
			var datasetBytesView = new Float32Array(imageData.width*imageData.height)
			
			for (var j=0; j<imageData.data.length; j++) {
				datasetBytesView[j] = imageData.data[j * 4] / 255;
			}

			for (var imageIndex=0; imageIndex<imageData.height; imageIndex+=BATCH_SIZE) {
				var labelIndex = imageIndex+(3000*i)
				var batch = await get_batch(BATCH_SIZE, datasetBytesView, trainLabels, imageIndex, labelIndex);
				const history = await model.fit(
				batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
				{batchSize: BATCH_SIZE, epochs: 1});
				const loss = history.history.loss[0];
				console.log(loss);
				losses.push(loss)
				plotGraph(losses)
				//showPredictions(testArray)
			}
		}
	}
}

train();