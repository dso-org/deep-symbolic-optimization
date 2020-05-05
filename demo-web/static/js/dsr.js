var done = false;
var bestFound = false;
var bestExprs = 0;
var step = 0;
var ajaxCall = 0;

divMainPlot = document.getElementById('main_plot');
divSubplot = document.getElementById('subplot');
divSubplot2 = document.getElementById('subplot2');

function resetPlot(graphDiv){
    /* remove all traces */
    while(graphDiv.data.length>0)
    {
        Plotly.deleteTraces(graphDiv, [0]);
    }
}

function resetButtons(){
    $('#btn_start').parent().show();
    $('#btn_step').parent().show();
    $('#btn_stop').parent().hide();
    $('#btn_pause').parent().hide();

}

var layout_fix_range = {
    'xaxis.autorange': false,
    'yaxis.autorange': false
    };

function plotDataPoints(){
    $.ajax({
        url: '/data_points',
        type: 'POST',
        data: 'training data',
        dataType: "json",
        success: function(response){
            console.log("Plot data points.");
            resetPlot(divMainPlot)
            Plotly.addTraces(divMainPlot, response);
            Plotly.relayout(divMainPlot, layout_fix_range)
        },
        error: function(error){
            console.log(error);
        }
    });
}

/* Plot data points */
$(function(){
    $('#btn_data').on('click', function(){
        plotDataPoints();
        return false;
    });
});

/* Stop running: Button STOP */
$(function(){
    $('#btn_stop').on('click', function(){
        done = true;
        
        resetButtons();
        plotDataPoints();

        bestFound = false;
        bestExprs = 0;
        step = 0;
        ajaxCall = 0;
    });
});

/* Pause running: Button PAUSE */
$(function(){
    $('#btn_pause').on('click', function(){
        done = true;
        
        $('#btn_start').parent().show();
        $(this).parent().hide();
        $(this).blur();
        // $(this).button('<span class="glyphicon glyphicon-play" aria-hidden="true"></span>RESUME')
        // change button -> RESUME
    });
});

function bringBestExpr(caller){
    $.ajax({
        url: '/main_lines',
        type: 'POST',
        data: JSON.stringify({step: step}),
        contentType: 'application/json;charset=UTF-8',
        dataType: "json",
        success: function(response){
            console.log(step)
            if (response.warn == true){
                console.log("End request: Data not uploaded.");
                alert("Please upload data. (Click upload button first)");
                resetButtons();
                done = true;
            } else {
                if (step%1000 == 0){
                    console.log(step)
                    console.log(ajaxCall)
                }
                
                if (response.update == true && done != true){
                    /* Plot new best expression */
                    bestFound = true;
                    // validness = Plotly.validate(response.plot)
                    // console.log(validness[0].msg)
                    var ii;
                    for (ii = 0; ii < divMainPlot.data.length-1; ii++){
                        Plotly.restyle(divMainPlot, {
                            'line.width': 2,
                            'line.color': '#000000',
                            opacity: Math.pow(0.8, divMainPlot.data.length-1-ii)
                        },ii)
                    }
                    
                    Plotly.addTraces(divMainPlot, JSON.parse(response.plot),divMainPlot.data.length-1);
                    
                    bestExprs++;
                }
                
                
                if (response.done == true){
                    done = true;
                    resetButtons();
                }
            };
        },
        error: function(error){
            console.log(error);
            resetButtons();
            // console.log("Stop requesting");
        }
    }).done(function(){
        if (caller == 'start'){
            if (done != true && ajaxCall < 300){
                // bringBestExpr();
                setTimeout(bringBestExpr.bind(null,'start'), 500);
                step += 10;
                ajaxCall++;
            }
        } else if (caller == 'step'){
            if (bestFound == false && done != true && ajaxCall < 300){
                // bringBestExpr();
                setTimeout(bringBestExpr.bind(null,'step'), 500);
                step += 10;
                ajaxCall++;
            }
        }
    });
}

/* Plot expression: Button STEP (Finds next best expression, then pause) */
$(function(){
    $('#btn_step').on('click', function(){
        $('#btn_stop').parent().show();
        $('#btn_start').parent().show();
        $(this).blur();

        done = false;
        bestFound = false;

        bringBestExpr('step');
        // change button -> RESUME
    });
});

/* Plot expression: Button START */
$(function(){
    $('#btn_start').on('click', function(){
        console.log("Start");
        
        /* change buttons */
        $('#btn_stop').parent().show();
        $('#btn_pause').parent().show();
        $(this).parent().hide();
        $(this).blur();

        /* continue plotting */
        // requestLoop:

        done = false;

        bringBestExpr('start');

        return false;
    });
});



var blankData = [{
    x: [0],
    y: [0],
    mode: 'marker',
    type: 'scatter'
}];

var layout = {
    autosize: false, /* long tick labels automatically grow the figure margins */
    margin: { 
        t: 50, 
        l: 50, 
        r: 50, 
        b: 50
    },
    hovermode: 'closest',
    config: { responsive: true }, // not working..
    // plot_bgcolor:"white",
    // paper_bgcolor:"white",
    showlegend: false,
    // legend: {orientation: "h"},
    // legend: {
    //     x: 1,
    //     xanchor: 'right',
    //     y: 1
    //   }
    xaxis: {
        showgrid: false,
        zeroline: false,
        showline: true,
        mirror: 'ticks',
        // gridcolor: '#bdbdbd',
        // gridwidth: 0.5,
        zerolinecolor: '#969696',
        zerolinewidth: 1,
        linecolor: '#636363',
        linewidth: 2,
        // range: [-2,2]
        autorange: true
      },
    yaxis: {
        showgrid: false,
        zeroline: false,
        showline: true,
        mirror: 'ticks',
        // gridcolor: '#bdbdbd',
        // gridwidth: 0.5,
        zerolinecolor: '#969696',
        zerolinewidth: 1,
        linecolor: '#636363',
        linewidth: 2,
        // range: [-2,2]
        autorange: true
    },
    title: {
        text: 'Best expression: ',
        xref: 'paper', 
        yref: 'paper', 
        x: 0, 
        y: 1.2,
        xanchor: 'left', 
        yanchor: 'bottom',
        font:{
            family: 'Arial',
            size: 15,
            color: 'rgb(37,37,37)'},
        showarrow: false
    },
    width: 800

};
Plotly.newPlot(divMainPlot, blankData, layout, {responsive: true});

var new_style = {
    'line.color': ['rgb(115,115,115)']
}
Plotly.restyle(divMainPlot, new_style)

/*** Subplots ***/
// not using plotly subplotting

var trace1 = {
    x: [0, 1, 2],
    y: [10, 11, 12],
    type: 'scatter'
};

var trace2 = {
    x: [2, 3, 4],
    y: [100, 110, 120],
    type: 'scatter'
};

var layoutSubplot = {
    autosize: false,
    margin: { 
        t: 0, 
        l: 80, 
        r: 100, 
        b: 10
    },
    legend: {orientation: "h"},
    xaxis: {
        automargin: true,
        title: {
            text: "Iterations",
            font:{
                size: 12
            }
        },
        tickfont: {
            // family: 'Old Standard TT, serif',
            size: 10,
            color: 'black'
        },
        showgrid: false,
        zeroline: false,
        showline: true,
        mirror: 'ticks',
        linecolor: '#636363',
        linewidth: 1.5
      },
    yaxis: {
        title: {
            text: "Reward",
            font:{
                size: 12
            },
        },
        tickfont: {
            // family: 'Old Standard TT, serif',
            size: 10,
            color: 'black'
        },
        showgrid: false,
        zeroline: false,
        showline: true,
        mirror: 'ticks',
        linecolor: '#636363',
        linewidth: 1.5
    //     // range: [-2,2]
    //     autorange: false
    },
    width: 580,
    height: 200
};

var layoutSubplot2 = {
    autosize: false,
    margin: { 
        t: 0, 
        l: 80, 
        r: 100, 
        b: 10
    },
    legend: {orientation: "h"},
    xaxis: {
        automargin: true,
        title: {
            text: "Reward",
            font:{
                size: 12
            }
        },
        tickfont: {
            // family: 'Old Standard TT, serif',
            size: 10,
            color: 'black'
        },
        showgrid: false,
        zeroline: false,
        showline: true,
        mirror: 'ticks',
        linecolor: '#636363',
        linewidth: 1.5
    //     range: [-2,2],
    //     autorange: false
      },
    yaxis: {
        title: {
            text: "Density",
            font:{
                size: 12
            }
        },
        tickfont: {
            // family: 'Old Standard TT, serif',
            size: 10,
            color: 'black'
        },
        showgrid: false,
        zeroline: false,
        showline: true,
        mirror: 'ticks',
        linecolor: '#636363',
        linewidth: 1.5
    //     // range: [-2,2]
    //     autorange: false
    },
    width: 580,
    height: 200
};

Plotly.newPlot(divSubplot, [trace1], layoutSubplot)
Plotly.newPlot(divSubplot2, [trace2], layoutSubplot2)


/* JS tooltip opt-in */
$(function () {
    $('[data-toggle="tooltip"]').tooltip()
  })



// #f7fcfd
// #e0ecf4
// #bfd3e6
// #9ebcda
// #8c96c6
// #8c6bb1
// #88419d
// #810f7c

// #fcfbfd
// #efedf5
// #dadaeb
// #bcbddc
// #9e9ac8
// #807dba
// #6a51a3
// #54278f
// #3f007d

// #ffffff
// #f0f0f0
// #d9d9d9
// #bdbdbd
// #969696
// #737373
// #525252
// #252525
// #000000


/*
TODO: model done -> reset numbers

Need to receive from server: 

- Fill in templates
    - best expression,
    - fitness,
    - iteration #

- data points: scatter

- new expression plot (add trace)

Server:
- Each plot trace w/ hover set

Front:
- static layout
    - bg, ticks, size, templates for current best
- update new scatters
- update old scatter colors
- interactions
    - test w/ step
    - play, pause, stop
    - add, remove

### diagnostics:
- axes match 
*/