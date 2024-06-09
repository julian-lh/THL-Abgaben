
function groundTruthFunction(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function addGaussianNoise(y, variance) {
    let standardDeviation = Math.sqrt(variance);
    let noise = 0;
    for (let i = 0; i < 12; i++) {
        noise += Math.random();
    }
    noise -= 6;
    noise *= standardDeviation;
    return y + noise;
}

function generateData(numSamples = 100, variance = 0.05) {
    let xValues = [];
    let yValues = [];
    let noisyYValues = [];

    // Erzeugen von N zufälligen x-Werten und Berechnen der y-Werte
    for (let i = 0; i < numSamples; i++) {
        let x = Math.random() * 4 - 2; // Zufälliger Wert im Intervall [-2, 2]
        let y = groundTruthFunction(x);
        let noisyY = addGaussianNoise(y, variance);

        xValues.push(x);
        yValues.push(y);
        noisyYValues.push(noisyY);
    }

    // Aufteilen der Daten in Trainings- und Testdaten
    let trainX = xValues.slice(0, numSamples / 2);
    let trainY = yValues.slice(0, numSamples / 2);
    let trainNoisyY = noisyYValues.slice(0, numSamples / 2);

    let testX = xValues.slice(numSamples / 2);
    let testY = yValues.slice(numSamples / 2);
    let testNoisyY = noisyYValues.slice(numSamples / 2);

    return {
        trainX,
        trainY,
        trainNoisyY,
        testX,
        testY,
        testNoisyY
    };
}