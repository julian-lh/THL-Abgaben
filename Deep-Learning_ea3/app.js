// app.js

async function loadModel() {
  const model = await tf.loadLayersModel('./tfjs_model/model.json');
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
  document.getElementById(
    "predictionResult"
  ).innerText = `Next word: ${nextWord}`;
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

// Main function to load model, word index, and set up event handlers
(async function main() {
  document.getElementById("statusMessage").innerText = "Loading model...";
  model = await loadModel();

  document.getElementById("statusMessage").innerText = "Loading word index...";
  const wordIndex = await loadWordIndex();

  vocabDict = wordIndex;
  vocabArray = Object.keys(wordIndex).sort((a, b) => wordIndex[a] - wordIndex[b]);

  document.getElementById("statusMessage").innerText = "Model is loaded and ready to use";
})();

// Export the functions to be used in HTML
export { predictWord, acceptWord, autoPredict, stopAutoPredict, reset };
