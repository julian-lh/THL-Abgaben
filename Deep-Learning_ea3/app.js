class Tokenizer {
    constructor() {
        this.wordIndex = {};
        this.indexWord = {};
        this.wordCount = 0;
    }

    fitOnTexts(texts) {
        texts.forEach((text) => {
            text.split(' ').forEach((word) => {
                if (!(word in this.wordIndex)) {
                    this.wordIndex[word] = this.wordCount;
                    this.indexWord[this.wordCount] = word;
                    this.wordCount++;
                }
            });
        });
    }

    textsToSequences(texts) {
        return texts.map((text) =>
            text.split(' ').map((word) => this.wordIndex[word] || 0)
        );
    }
}

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
    const vocabSize = tokenizer.wordCount + 1;
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

    const xsDataset = tf.data.array(xsTensor.arraySync());
    const ysDataset = tf.data.array(ysTensor.arraySync());

    const dataset = tf.data.zip({ xs: xsDataset, ys: ysDataset }).batch(32).prefetch(32);
    return dataset;
}


function createModel(vocabSize, nSteps = 5) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 100, inputLength: nSteps }));
    model.add(tf.layers.lstm({ units: 100, activation: 'relu', returnSequences: true }));
    model.add(tf.layers.lstm({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.01),
        metrics: ['accuracy']
    });

    return model;
}

async function trainModel(model, dataset, epochs = 10) {
    const datasetArray = await dataset.toArray();
    const valSize = Math.floor(0.2 * datasetArray.length);
    const trainSize = datasetArray.length - valSize;

    const trainDataset = tf.data.array(datasetArray.slice(0, trainSize)).batch(32);
    const valDataset = tf.data.array(datasetArray.slice(trainSize)).batch(32);

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

async function loadModel() {
    const model = await tf.loadLayersModel('./tfjs_model/model.json');
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.01),
        metrics: ['accuracy']
    });
    return model;
}

async function loadWordIndex() {
    const response = await fetch('./word_index.json');
    const wordIndex = await response.json();
    return wordIndex;
}

let vocabDict, vocabArray, model;

function predictNextWord(model, input, vocabArray) {
    const inputTensor = tf.tensor2d([input], [1, input.length]);
    const prediction = model.predict(inputTensor);
    const predictedIndex = prediction.argMax(1).dataSync()[0];
    return vocabArray[predictedIndex];
}

function predictWord() {
    const inputText = document.getElementById("inputText").value.trim();
    const tokens = inputText.split(" ");
    const inputIndices = tokens.map((word) => vocabDict[word]);

    const nextWord = predictNextWord(model, inputIndices, vocabArray);
    document.getElementById("predictionResult").innerText = `Next word: ${nextWord}`;
}

function acceptWord() {
    const inputText = document.getElementById("inputText").value.trim();
    const tokens = inputText.split(" ");
    const inputIndices = tokens.map((word) => vocabDict[word]);

    const nextWord = predictNextWord(model, inputIndices, vocabArray);
    document.getElementById("inputText").value += ` ${nextWord}`;
}

function autoPredict() {
    let count = 0;
    const interval = setInterval(() => {
        if (count >= 10) {
            clearInterval(interval);
            return;
        }
        acceptWord();
        count++;
    }, 500);
}

function stopAutoPredict() {
    clearInterval(interval);
}

function reset() {
    document.getElementById("inputText").value = "";
    document.getElementById("predictionResult").innerText = "";
}

async function saveModel() {
    await model.save('downloads://model');
    const wordIndexBlob = new Blob([JSON.stringify(vocabDict)], { type: 'application/json' });
    const url = URL.createObjectURL(wordIndexBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'word_index.json';
    a.click();
}

(async function main() {
    // document.getElementById("statusMessage").innerText = "Loading model...";
    // model = await loadModel();

    // document.getElementById("statusMessage").innerText = "Loading word index...";
    // const wordIndex = await loadWordIndex();

    // vocabDict = wordIndex;
    // vocabArray = Object.keys(wordIndex).sort((a, b) => wordIndex[a] - wordIndex[b]);

    // document.getElementById("statusMessage").innerText = "Model is loaded and ready to use";
})();
