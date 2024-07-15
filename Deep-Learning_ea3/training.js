async function loadData(url) {
    const response = await fetch(url);
    const text = await response.text();
    const sentences = text.split('. ');
    return sentences;
}

function preprocessData(sentences) {
    const tokenizer = new Tokenizer();
    tokenizer.fitOnTexts(sentences);
    const sequences = tokenizer.textsToSequences(sentences);
    const vocabSize = tokenizer.wordIndex.size + 1;
    return { sequences, tokenizer, vocabSize };
}

function prepareDataForTraining(sequences, vocabSize, nSteps = 5) {
    const xs = [];
    const ys = [];

    sequences.forEach(seq => {
        for (let i = nSteps; i < seq.length; i++) {
            xs.push(seq.slice(i - nSteps, i));
            ys.push(seq[i]);
        }
    });

    const xsTensor = tf.tensor2d(xs, [xs.length, nSteps]);
    const ysTensor = tf.tensor1d(ys, 'int32');

    const dataset = tf.data.zip({ xs: xsTensor, ys: ysTensor }).batch(32).prefetch(tf.data.AUTOTUNE);
    return dataset;
}

function createModel(vocabSize, nSteps = 5) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 100, inputLength: nSteps }));
    model.add(tf.layers.lstm({ units: 100, activation: 'relu', returnSequences: true }));
    model.add(tf.layers.lstm({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

    model.compile({
        loss: 'sparse_categorical_crossentropy',
        optimizer: tf.train.adam(0.01),
        metrics: ['accuracy']
    });

    return model;
}

async function trainModel(model, dataset, epochs = 10) {
    const datasetSize = await dataset.size();
    const valSize = Math.floor(0.2 * datasetSize);
    const trainSize = datasetSize - valSize;

    const trainDataset = dataset.take(trainSize);
    const valDataset = dataset.skip(trainSize);

    const history = await model.fitDataset(trainDataset, {
        epochs,
        validationData: valDataset
    });

    return history;
}

async function saveModel(model, wordIndex) {
    await model.save('downloads://model');
    const wordIndexBlob = new Blob([JSON.stringify(wordIndex)], { type: 'application/json' });
    const url = URL.createObjectURL(wordIndexBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'word_index.json';
    a.click();
}

async function startTraining() {
    const sentences = await loadData('metamorphosis_clean.txt');
    const { sequences, tokenizer, vocabSize } = preprocessData(sentences);
    const dataset = prepareDataForTraining(sequences, vocabSize);
    const model = createModel(vocabSize);
    await trainModel(model, dataset);
    await saveModel(model, tokenizer.wordIndex);
}
