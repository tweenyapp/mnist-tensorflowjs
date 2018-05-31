//import * as tf from '@tensorflow/tfjs';


var IMAGE_SIZE = 784;
var NUM_CLASSES = 10;
var NUM_DATASET_ELEMENTS = 60000;

var NUM_TEST_ELEMENTS = 10000;

function load(path) {
	var img = new Image();
	var imgRequest = new Promise(resolve => {
	img.crossOrigin = "Anonymous";
	img.onload = function() {
		var canvas = document.createElement('canvas');
		var ctx = canvas.getContext('2d');
		canvas.width = img.width;
    	canvas.height = img.height;
    	ctx.drawImage(img, 0, 0);
    	var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		resolve(imageData);
    	}
	});
	img.src = path
	return imgRequest
}

var show = function(image, label, prediction) {
	var width = 28,
		height = 28

	var canvas = document.createElement('canvas'),
	ctx = canvas.getContext('2d');

	//console.log(label, prediction)
	canvas.width = width;
	canvas.height = height;

	// create imageData object
	var idata = ctx.createImageData(width, height);

	// set our buffer as source
	idata.data.set(image);

	// update canvas with new data
	ctx.putImageData(idata, 0, 0);
	var dataUri = canvas.toDataURL();
	var img = document.getElementById("mnist");
	img.src =  dataUri;
	document.getElementById('label').innerHTML = label;
	document.getElementById('prediction').innerHTML = prediction;
}

function get_batch(batchSize, datasetBytesView, mnistLabels, imageIndex, labelIndex) {
	var imageBatch = new Float32Array(batchSize*IMAGE_SIZE)
	imageBatch = datasetBytesView.slice(imageIndex*IMAGE_SIZE,(imageIndex+batchSize)*IMAGE_SIZE)
	const xs = tf.tensor2d(imageBatch, [batchSize, IMAGE_SIZE]);
	const labels = tf.oneHot(tf.tensor1d(mnistLabels.slice(labelIndex,labelIndex+batchSize), dtype='int32'), 10)
	return {xs, labels};
}

async function visualise(){
	for (var i=0; i<20; i++) {
		
		var imageData = await load('http://localhost/images/mnist_batch_' + i +'.png');
		
		for (var index=0; index<imageData.height; index++) {
			var image = imageData.data.slice(index*784*4,784*4*(index+1))
			await show(image)
			console.log(trainLabels[index])
		}
	}
}

//visualise();