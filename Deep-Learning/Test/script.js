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

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    return model;
}

async function trainModel(model, trainX, trainY, epochs = 100) {
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;

    return await model.fit(trainX, trainY, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

async function evaluateModel(model, testX, testY) {
    const result = model.evaluate(testX, testY);
    const loss = result[0].dataSync()[0];
    const mse = result[1].dataSync()[0];
    return { loss, mse };
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

async function run() {
    const { X: trainX, Y: trainY } = await generateData(50);
    const { X: testX, Y: testY } = await generateData(50);

    plotData(trainX, trainY, testX, testY);

    const model = createModel();
    await trainModel(model, trainX, trainY, 100);

    const { loss, mse } = await evaluateModel(model, testX, testY);
    document.getElementById('evaluation-results').innerHTML = `
        <p>Test Loss: ${loss.toFixed(4)}</p>
        <p>Test MSE: ${mse.toFixed(4)}</p>
    `;

    await model.save('downloads://my-model');
}

run();
