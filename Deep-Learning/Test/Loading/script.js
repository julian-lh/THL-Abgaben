async function generateData(N, noiseVar = 0.05) {
    const X = [];
    const Y = [];
    for (let i = 0; i < N; i++) {
        const x = Math.random();
        const noise = Math.sqrt(noiseVar) * tf.randomNormal([1]).dataSync()[0];
        const y = 2 * x + 1 + noise;  // Example linear relationship with noise
        X.push(x);
        Y.push(y);
    }
    return { X: tf.tensor2d(X, [N, 1]), Y: tf.tensor2d(Y, [N, 1]) };
}

async function loadModel(modelUrl) {
    try {
        const model = await tf.loadLayersModel(modelUrl);
        // Compile the model after loading
        model.compile({
            optimizer: tf.train.adam(0.01),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse'],
        });
        return model;
    } catch (error) {
        console.warn(`Model at ${modelUrl} not found`);
        return null;
    }
}

function plotData(trainX, trainY, testX, testY) {
    const series = ['Train Data', 'Test Data'];
    const trainData = Array.from(trainX.dataSync()).map((x, i) => ({ x, y: trainY.dataSync()[i] }));
    const testData = Array.from(testX.dataSync()).map((x, i) => ({ x, y: testY.dataSync()[i] }));

    tfvis.render.scatterplot(
        { name: 'Data Plot', tab: 'Data' },
        { values: [trainData, testData], series },
        { xLabel: 'X', yLabel: 'Y', height: 300 }
    );
}

async function evaluateModel(model, testX, testY) {
    const result = model.evaluate(testX, testY);
    const loss = result[0].dataSync()[0];
    const mse = result[1].dataSync()[0];
    return { loss, mse };
}

function plotPredictions(model, testX, testY) {
    const testPredictions = model.predict(testX);
    const predictionsData = Array.from(testX.dataSync()).map((x, i) => ({
        x,
        y: testPredictions.dataSync()[i],
    }));
    const testData = Array.from(testX.dataSync()).map((x, i) => ({
        x,
        y: testY.dataSync()[i],
    }));

    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Test Data', tab: 'Evaluation' },
        { values: [testData, predictionsData], series: ['Test Data', 'Predictions'] },
        { xLabel: 'X', yLabel: 'Y', height: 300 }
    );
}

async function run() {
    const { X: trainX, Y: trainY } = await generateData(50);
    const { X: testX, Y: testY } = await generateData(50);

    plotData(trainX, trainY, testX, testY);

    const modelUrl = './THL-Abgaben/Deep-Learning/Test/Loading/my-model.json'; // Update this path to your actual model path
    const model = await loadModel(modelUrl);

    if (model) {
        const { loss, mse } = await evaluateModel(model, testX, testY);
        document.getElementById('evaluation-results').innerHTML = `
            <p>Test Loss: ${loss.toFixed(4)}</p>
            <p>Test MSE: ${mse.toFixed(4)}</p>
        `;

        plotPredictions(model, testX, testY);
    } else {
        document.getElementById('evaluation-results').innerHTML = `
            <p>Model could not be loaded.</p>
        `;
    }
}

run();
