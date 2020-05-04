var done = true;
var step = 0;
var ajaxCall = 0;

divMainPlot = document.getElementById('main_plot');

function resetPlot(graphDiv){
    /* remove all traces */
    while(graphDiv.data.length>0)
    {
        Plotly.deleteTraces(graphDiv, [0]);
    }
}

function resetButtons(){
    $('#btn_start').parent().show();
    $('#btn_stop').parent().hide();
    $('#btn_pause').parent().hide();
}

var layout_fix_range = {
    'xaxis.autorange': false,
    'yaxis.autorange': false
    };

/* Plot data points */
$(function(){
    $('#btn_data').on('click', function(){
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
        return false;
    });
});

/* Stop running */
$(function(){
    $('#btn_stop').on('click', function(){
        done = true;
        step = 0;
        resetButtons();
    });
});
/* Pause running */
$(function(){
    $('#btn_pause').on('click', function(){
        done = true;
        $(this).button('<span class="glyphicon glyphicon-play" aria-hidden="true"></span>RESUME')
    });
});

function bringBestExpr(){
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
                
                if (response.update == true){
                    /* Plot new best expression */
                    // validness = Plotly.validate(response.plot)
                    // console.log(validness[0].msg)
                    Plotly.addTraces(divMainPlot, JSON.parse(response.plot));
                    // restyle existing plots
                    // Div.data.length
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
            // console.log("stop requesting");
            // break;
            // break requestLoop;
        }
    }).done(function(){
        if (done != true && ajaxCall < 300){
            // bringBestExpr();
            setTimeout(bringBestExpr, 500);
            step += 10;
            ajaxCall++;
            }
    });
}

/* Plot expression */
$(function(){
    $('#btn_start').on('click', function(){
        console.log("Start");
        
        /* change buttons */
        $('#btn_stop').parent().show();
        $('#btn_pause').parent().show();
        $(this).parent().hide();
        $(this).blur();
        
        done = false;
        step = 0;
        bestExprs = 0;

        ajaxCall = 0;

        /* continue plotting */
        // requestLoop:
        bringBestExpr();

        return false;
    });
});



var blank_data = [{
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
        b: 100
    },
    hovermode: 'closest',
    config: { responsive: true }, // re
    // plot_bgcolor:"white",
    // paper_bgcolor:"white",
    // showlegend: false,
    legend: {"orientation": "h"},
    // legend: {
    //     x: 1,
    //     xanchor: 'right',
    //     y: 1
    //   }
    xaxis: {
        showgrid: false,
        zeroline: true,
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
        zeroline: true,
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
    title: 'Title'

};
Plotly.newPlot(divMainPlot, blank_data, layout, {responsive: true});

var new_style = {
    'line.color': ['rgb(115,115,115)']
}
Plotly.restyle(divMainPlot, new_style)

var new_best_exp = 'sinx'

var update_best_info = {
    title: 'Best expression: ' + new_best_exp
}




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