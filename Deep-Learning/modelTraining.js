function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // Adding hidden layers
    model.add(tf.layers.dense({ units: 100, activation: 'ReLU' }));
    model.add(tf.layers.dense({ units: 100, activation: 'ReLU' }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    return model;
}

function createModel2() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // Adding hidden layers
    model.add(tf.layers.dense({ units: 100, activation: 'ReLU' }));
    model.add(tf.layers.dense({ units: 100, activation: 'ReLU' }));
    model.add(tf.layers.dense({ units: 100, activation: 'ReLU' }));
    model.add(tf.layers.dense({ units: 100, activation: 'ReLU' }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    return model;
}

async function trainModel(model, inputs, labels, epochs = 80) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    // const epochs = 100;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

async function convertAndTrainModel(model, inputData, labelData, epochs = 80) {

    originalData = inputData.map((xValue, index) => ({
        x: xValue,
        y: labelData[index]
    }));

    const { inputs, labels } = convertToTensor(originalData);

    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    // const epochs = 100;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.x)
        const labels = data.map(d => d.y);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        return {
            inputs: inputTensor,
            labels: labelTensor
        }
    });
}


function getPrediction(model, x) {
    const xs = tf.tensor2d(x, [x.length, 1]);
    const ys = model.predict(xs);
    return ys.dataSync();
}

function evaluateModel(model, inputs, labels) {
    const tensorInputs = tf.tensor2d(inputs, [inputs.length, 1]);
    const tensorLabels = tf.tensor2d(labels, [labels.length, 1]);
    const evaluation = model.evaluate(tensorInputs, tensorLabels);
    return { loss: evaluation[0].dataSync(), mse: evaluation[1].dataSync() };
}