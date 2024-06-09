
function drawPlotOld(div_id, x, y) {
    DATASET = document.getElementById(div_id);
    Plotly.newPlot(DATASET, [{
        x: x,
        y: y
    }], {
        margin: { t: 0 }
    });
}

function drawPlot(div_id, x, y, label = '', color = 'blue') {
    DATASET = document.getElementById(div_id);

    var trace = [{
        x: x,
        y: y,
        mode: 'markers',
        type: 'scatter',
        name: label,
        marker: { size: 6, color: color }
    }];

    Plotly.newPlot(DATASET, trace);
}

function drawPairPlot(div_id, x1, y1, x2, y2, label1 = '', label2 = '') {
    DATASET = document.getElementById(div_id);

    var trace1 = {
        x: x1,
        y: y1,
        mode: 'markers',
        type: 'scatter',
        name: label1,
        marker: { size: 6, color: 'blue' }
    };

    var trace2 = {
        x: x2,
        y: y2,
        mode: 'markers',
        type: 'scatter',
        name: label2,
        marker: { size: 6, color: 'orange' }
    };
    var data = [trace1, trace2]

    Plotly.newPlot(DATASET, data);
}