async function loadData() {
    // Load CSV data
    const df = await dfd.readCSV('path/to/your/dataset.csv');
    return df;
}

function preprocessData(df) {
    // Assuming the column with text data is named 'text'
    const text = df['text'].values;

    // Tokenize the text data
    const tokens = text.map(sentence => sentence.split(' '));

    // Create a dictionary and convert tokens to indices
    const vocab = new Set(tokens.flat());
    const vocabArray = Array.from(vocab);
    const vocabDict = vocabArray.reduce((acc, word, idx) => {
        acc[word] = idx;
        return acc;
    }, {});

    const sequences = tokens.map(sentence => sentence.map(word => vocabDict[word]));
    return { sequences, vocabDict, vocabArray };
}

// Create LSTM Model
function createModel(vocabSize) {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 100, returnSequences: true, inputShape: [null, vocabSize] }));
    model.add(tf.layers.lstm({ units: 100 }));
    model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));
    return model;
}

// Train the model
async function trainModel(model, data, labels, epochs = 10) {
    await model.fit(data, labels, {
        epochs: epochs,
        batchSize: 32,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'loss' })
    });
}

// Predict next word
function predictNextWord(model, input, vocabArray) {
    const inputTensor = tf.tensor2d([input]);
    const prediction = model.predict(inputTensor);
    const predictedIndex = prediction.argMax(1).dataSync()[0];
    return vocabArray[predictedIndex];
}

function predictWord() {
    const inputText = document.getElementById("inputText").value.trim();
    const tokens = inputText.split(' ');
    const inputIndices = tokens.map(word => vocabDict[word]);

    const nextWord = predictNextWord(model, inputIndices, vocabArray);
    document.getElementById("predictionResult").innerText = `Next word: ${nextWord}`;
}

function acceptWord() {
    const inputText = document.getElementById("inputText").value.trim();
    const tokens = inputText.split(' ');
    const inputIndices = tokens.map(word => vocabDict[word]);

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
    document.getElementById("inputText").value = '';
    document.getElementById("predictionResult").innerText = '';
}

// Main function to load data, preprocess it, and train the model
(async function main() {
    const df = await loadData();
    const { sequences, vocabDict, vocabArray } = preprocessData(df);

    const vocabSize = vocabArray.length;

    // Prepare data for training
    const inputData = tf.tensor2d(sequences.slice(0, -1));
    const labelsData = tf.tensor2d(sequences.slice(1));

    const model = createModel(vocabSize);
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy'
    });

    await trainModel(model, inputData, labelsData, 10);

    // Example prediction
    const exampleInput = sequences[0];
    const nextWord = predictNextWord(model, exampleInput, vocabArray);
    console.log("Predicted next word:", nextWord);
})();

