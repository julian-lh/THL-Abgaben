const noiseVariance = 0.05;
const numSamples = 100;

function yFunction(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateData() {
    const xValues = [];
    const yValues = [];
    for (let i = 0; i < numSamples; i++) {
        const x = Math.random() * 4 - 2;
        const y = yFunction(x);
        xValues.push(x);
        yValues.push(y);
    }
    return { xValues, yValues };
}

function splitData(xValues, yValues) {
    const indices = tf.util.createShuffledIndices(numSamples);
    const trainSize = numSamples / 2;
    const xTrain = [];
    const yTrain = [];
    const xTest = [];
    const yTest = [];

    for (let i = 0; i < trainSize; i++) {
        xTrain.push(xValues[indices[i]]);
        yTrain.push(yValues[indices[i]]);
    }
    for (let i = trainSize; i < numSamples; i++) {
        xTest.push(xValues[indices[i]]);
        yTest.push(yValues[indices[i]]);
    }
    return { xTrain, yTrain, xTest, yTest };
}

function addNoise(yValues) {
    return yValues.map(y => y + tf.randomNormal([1], 0, Math.sqrt(noiseVariance)).arraySync()[0]);
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    return model;
}

async function trainModel(model, xTrain, yTrain, epochs, xTest, yTest) {
    return await model.fit(xTrain, yTrain, {
        epochs,
        validationData: [xTest, yTest],
        batchSize: 32,
        verbose: 0
    });
}

function plotData(divId, xTrain, yTrain, xTest, yTest, title) {
    const traceTrain = {
        x: xTrain,
        y: yTrain,
        mode: 'markers',
        type: 'scatter',
        name: 'Train',
        marker: { color: 'blue' }
    };

    const traceTest = {
        x: xTest,
        y: yTest,
        mode: 'markers',
        type: 'scatter',
        name: 'Test',
        marker: { color: 'red' }
    };

    const layout = {
        title: title,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    };

    Plotly.newPlot(divId, [traceTrain, traceTest], layout);
}

function plotPredictions(divId, model, xData, yData, title, mse, loss) {
    const preds = model.predict(tf.tensor2d(xData, [xData.length, 1])).arraySync();

    const traceActual = {
        x: xData,
        y: yData,
        mode: 'markers',
        type: 'scatter',
        name: 'Actual',
        marker: { color: 'blue' }
    };

    const tracePred = {
        x: xData,
        y: preds.map(p => p[0]),
        mode: 'markers',
        type: 'scatter',
        name: 'Prediction',
        line: { color: 'red' }
    };

    const layout = {
        title: `${title} (Loss: ${loss.toFixed(4)}, MSE: ${mse.toFixed(4)})`,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    };

    Plotly.newPlot(divId, [traceActual, tracePred], layout);
}

async function run() {
    const { xValues, yValues } = generateData();
    const { xTrain, yTrain, xTest, yTest } = splitData(xValues, yValues);
    const yTrainNoisy = addNoise(yTrain);
    const yTestNoisy = addNoise(yTest);

    const xTrainTensor = tf.tensor2d(xTrain, [xTrain.length, 1]);
    const yTrainTensor = tf.tensor2d(yTrain, [yTrain.length, 1]);
    const yTrainNoisyTensor = tf.tensor2d(yTrainNoisy, [yTrainNoisy.length, 1]);

    const xTestTensor = tf.tensor2d(xTest, [xTest.length, 1]);
    const yTestTensor = tf.tensor2d(yTest, [yTest.length, 1]);
    const yTestNoisyTensor = tf.tensor2d(yTestNoisy, [yTestNoisy.length, 1]);

    // Plotten
    plotData('data-plot-clean', xTrain, yTrain, xTest, yTest, '(Clean)');
    plotData('data-plot-noisy', xTrain, yTrainNoisy, xTest, yTestNoisy, '(Noisy)');

    // Training clean Daten
    const modelClean = createModel();
    const historyClean = await trainModel(modelClean, xTrainTensor, yTrainTensor, 100, xTestTensor, yTestTensor);
    const mseCleanTrain = historyClean.history.loss[historyClean.history.loss.length - 1];
    const mseCleanTest = historyClean.history.val_loss[historyClean.history.val_loss.length - 1];
    plotPredictions('prediction-clean-train', modelClean, xTrain, yTrain, '(Clean Training)', mseCleanTrain, mseCleanTrain);
    plotPredictions('prediction-clean-test', modelClean, xTest, yTest, '(Clean Test)', mseCleanTest, mseCleanTest);

    // Training noisy Daten (Best-Fit)
    const modelNoisyBestFit = createModel();
    const historyNoisyBestFit = await trainModel(modelNoisyBestFit, xTrainTensor, yTrainNoisyTensor, 50, xTestTensor, yTestNoisyTensor);
    const mseNoisyBestFitTrain = historyNoisyBestFit.history.loss[historyNoisyBestFit.history.loss.length - 1];
    const mseNoisyBestFitTest = historyNoisyBestFit.history.val_loss[historyNoisyBestFit.history.val_loss.length - 1];
    plotPredictions('prediction-best-train', modelNoisyBestFit, xTrain, yTrainNoisy, '(Best-Fit Training)', mseNoisyBestFitTrain, mseNoisyBestFitTrain);
    plotPredictions('prediction-best-test', modelNoisyBestFit, xTest, yTestNoisy, '(Best-Fit Test)', mseNoisyBestFitTest, mseNoisyBestFitTest);

    // Training noisy Daten (Over-Fit)
    const modelNoisyOverFit = createModel();
    const historyNoisyOverFit = await trainModel(modelNoisyOverFit, xTrainTensor, yTrainNoisyTensor, 200, xTestTensor, yTestNoisyTensor);
    const mseNoisyOverFitTrain = historyNoisyOverFit.history.loss[historyNoisyOverFit.history.loss.length - 1];
    const mseNoisyOverFitTest = historyNoisyOverFit.history.val_loss[historyNoisyOverFit.history.val_loss.length - 1];
    plotPredictions('prediction-overfit-train', modelNoisyOverFit, xTrain, yTrainNoisy, '(Over-Fit Training)', mseNoisyOverFitTrain, mseNoisyOverFitTrain);
    plotPredictions('prediction-overfit-test', modelNoisyOverFit, xTest, yTestNoisy, '(Over-Fit Test)', mseNoisyOverFitTest, mseNoisyOverFitTest);
}

run();
