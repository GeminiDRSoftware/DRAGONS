// ViewPort parent class
function ViewPort(element, id) {
    if (element && id) {
	this.element = element; 
	this.id = id;
	this.init();
    }
}
ViewPort.prototype = {
    constructor: ViewPort,
    init: function() {
	var html_str = this.composeHTML();
	this.element.html(html_str);
    }, // end init
    composeHTML: function() {
	return '<div id='+this.id+' class="view_port"></div>';
    }, // end composeHTML
    addRecord: function(records) {
	if (!(records instanceof Array)) {
	    records = [records];
	}
	var record_str = "";
	for (var i in records) {
	    record_str += '<span>'+records[i]+'</span>';
	}
	$('#'+this.id).append(record_str);
    },
    clearRecord: function() {
	$('#'+this.id).html("");
    }
};


// ScrollTable child class
// constructor
// element should be a jQuery selection of an empty DOM element (ie. div),
// id is a string, and columns is an object containing column model information
function ScrollTable(element,id,columns) {
    this.element = element;
    this.id = id;
    this.columns = columns;
    this.rows = {};
    this.records = {};
    this.init();
}
// prototype: inherit from ViewPort
var st_proto = new ViewPort();
ScrollTable.prototype = st_proto;
ScrollTable.prototype.constructor = ScrollTable;

// add ScrollTable-specific methods
ScrollTable.prototype.init = function() {

    // Add an empty div to the element
    var html_str = this.composeHTML();
    this.element.html(html_str);

    // Add an event handler to the headers of swappable columns
    // to do the swap
    var st = this;
    this.element.on("click","th.swap",function() {
        for (var col in st.columns) {
	    col = st.columns[col];
	    if ($(this).hasClass(col.id)) {

		// Swap alt and std fields/names
		var temp = col.name;
		col.name = col.alt_name;
		col.alt_name = temp;
		temp = col.field;
		col.field = col.swap;
		col.swap = temp;

		// Put new name in header (with swap icon)
		$(this).html('<div style="position:relative">'+
			     col.name+
			     '<span class="swap_icon"></span>'+
			     '</div>');

		    
		// Put alternate data in data cells
		$("#"+st.id+" td."+col.id).each(function(){
		    var key = $(this).attr("id");
		    if (key==undefined) {
			key = $(this).parent().attr("id");
		    }
		    var data = st.records[key][col.field];
		    $(this).text(data);
		});

		break;
	    }
	}
	// Throw a custom event, as a hook for the manager
	// to do something (eg. modify a plot)
	$(this).trigger("swapColumn");

    }); // end on click

}; // end init

ScrollTable.prototype.composeHTML = function() {
    // Compose table tags with column headers
    var html_str = '<table class="scroll_table" id="'
                   +this.id+'"><thead class="fixed_header">';
    html_str += '<tr style="position:relative;display:block">'
    for (i in this.columns) {
	col = this.columns[i];
	if (col.hidden) {
	    continue;
	}


	// Add classes for column id, as well swap property
	html_str += '<th  class="'+col.id;
	var w;
	if (col.swap) {
	    html_str += ' swap';
	}
	html_str += '"';


	// Set column width
	// Add 16px to the width of the last header element for the scroll bar
	w = col.width;
	if (i==this.columns.length-1) {
	    w+=16;
	}
	html_str += ' width='+w+'px';
	html_str += '>';

	if (col.swap) {
	    html_str += '<div style="position:relative">'+
	                col.name+
		        '<span class="swap_icon"></span>'+
		        '</div>';
	} else {
	    html_str += col.name;
	}
	
	html_str += '</th>';
    }
    html_str += '</tr></thead>';

    // Add table body
    html_str += '<tbody class="scroll_body" width="100%" '+
                'height="'+ 
                (parseInt(this.element.height())-32) +'px"' +
	        'style="display:block;overflow:auto;position:relative">';

    html_str += '</tbody>';

    html_str += '</table>';

    return html_str;
}; // end composeHTML

ScrollTable.prototype.addRecord = function(records) {
    // Make records into an array if it is not already
    if (!(records instanceof Array)) {
	records = [records];
    }

    // Loop through records, making new table rows and
    // adding them to the tbody
    var tbody = $('#'+this.id+' tbody');
    for (var i in records) {

	var record = records[i];

	// Add the record to the stored records object
	this.records[record['key']] = record;

	// Generate a table row from input data
	var table_row = "";

	// Give it an id corresponding to the key specified in the record
	table_row += '<tr id="'+record['key']+'">';

	var sort_col = null;
	for (var j in this.columns) {
	    col = this.columns[j];
	    if (col.sort && !sort_col) {
		sort_col = col;
	    }
	    if (!col.hidden) {
		table_row += '<td class="'+col.id+
		             '" width="'+col.width+'px">'+
		             record[col.field]+'</td>';
	    }
	}
	table_row += '</tr>';
	
	// Add the new row to the database of rows
	if ($("#"+record['key']).length>0) {
	    // A row with this key exists already, remove it
	    $("#"+record['key']).remove();
	}
	if (sort_col) {
	    var rec = this.records;
	    placed = false;

	    // Iterate backward over the rows, assuming that
	    // data is likely to come in mostly in order
	    var backward_rows = tbody.find("tr").get().reverse();
	    for (var k in backward_rows) {
		var this_row = $(backward_rows[k]);
	        var this_field = rec[this_row.attr("id")][sort_col.field];
		if (record[sort_col.field]>=this_field) {
		    $(table_row).insertAfter(this_row);
		    placed = true;
		    break;
		}
	    }
	    if (!placed) {
		tbody.prepend(table_row);
	    }
	} else {
	    // No sorting, just append it to the tbody
	    tbody.append(table_row);
	}

	// Update the rows database with this row
	this.rows[record['key']] = table_row;
    }

    // Check for overflow: if none, add 16px to all elements
    // in the  last table column 
    var last_col = this.columns[this.columns.length-1];
    if (tbody[0].clientHeight==tbody[0].scrollHeight) {
	$('#'+this.id+' td.'+last_col.id).attr(
	    "width",(last_col.width+16)+"px");
    } else {
	$('#'+this.id+' td.'+last_col.id).attr(
	    "width",last_col.width+"px");
    }

}; // end addRecord

ScrollTable.prototype.clearRecord = function() {
    // Clear out the tbody, leaving the thead alone
    $('#'+this.id+' tbody').html("");

    // Reset records and rows objects
    this.records = {};
    this.rows = {};
}; // end clearRecord



// TimePlot child class
// constructor
// element should be a jQuery selection of an empty DOM element (ie. div),
// id is a string, and columns is an object containing column model information
function TimePlot(element,id,options) {
    this.element = element;
    this.id = id;
    this.options = options;
    this.plot = null;
    this.selected = {};
    this.series_index = {};
    this.ut = false;
    this.ut_offset = 0;
    this.timezone = "";
    this.init();
}
// prototype: inherit from ViewPort
var tp_proto = new ViewPort();
TimePlot.prototype = tp_proto;
TimePlot.prototype.constructor = TimePlot;

// add TimePlot-specific methods
TimePlot.prototype.init = function(record) {

    // Add an empty div to the element
    var html_str = this.composeHTML();
    this.element.html(html_str);

    // Create an empty jqPlot in the element
    var data = [];
    var series_options = [];
    var series_to_use = this.options.series_labels;
    for (var i in series_to_use) {
	var series = series_to_use[i];

	// Keep the series index in a dictionary
	this.series_index[series] = i;

	// Initialize with all series selected for display
	this.selected[series_to_use[i]] = true;

	// Make empty data structure for each series
	data.push([null]);	

	// Set up series labels and band data
	series_options.push({label:series,
		             showLabel:false,
		             rendererOptions: {bandData:[[null],[null]],
			     highlightMouseOver:false}});
    }

    // Set up legend options
    var legend_options = {};
    if (series_to_use.length>1) {
	legend_options = {show: true,
			  renderer: $.jqplot.EnhancedLegendRenderer,
			  rendererOptions: {seriesToggle:true,
	                                    numberRows:1},
			  preDraw: true,
			  location: "ne",
			  marginTop:0,
			  marginRight:0,
			  placement: "insideGrid"};
    }

    // Check whether date should be displayed in UT
    var date = new $.jsDate;
    this.ut_offset = date.getUtcOffset();
    this.timezone = date.getTimezoneAbbr();
    var mindate, maxdate;
    if (this.options.ut && !this.ut) {
	mindate = new $.jsDate(this.options.mindate);
	mindate.add(this.ut_offset,"milliseconds");
	this.options.mindate = mindate.getTime();

	maxdate = new $.jsDate(this.options.maxdate);
	maxdate.add(this.ut_offset,"milliseconds");
	this.options.maxdate = maxdate.getTime();

	this.ut = true;
    } else if (!this.options.ut && this.ut) {
	mindate = new $.jsDate(this.options.mindate);
	mindate.add(-this.ut_offset,"milliseconds");
	this.options.mindate = mindate.getTime();

	maxdate = new $.jsDate(this.options.maxdate);
	maxdate.add(-this.ut_offset,"milliseconds");
	this.options.maxdate = maxdate.getTime();

	this.ut = false;
    } else {
	mindate = new $.jsDate(this.options.mindate);
	maxdate = new $.jsDate(this.options.maxdate);
	this.options.mindate = mindate.getTime();
	this.options.maxdate = maxdate.getTime();
    }    

    if (this.ut) {
	this.options.xaxis_label += " (UT)";
    } else {
	this.options.xaxis_label += " ("+this.timezone+")";
    }

    // Compile all jqPlot configuration options
    this.config = { "title":this.options.title,
		    axes: { xaxis: {renderer:$.jqplot.DateAxisRenderer,
				    tickOptions:{formatString:"%H:%M",
						 angle:-30,
						 fontSize:"8pt"},
				    tickInterval:"1 hour",
				    tickRenderer:$.jqplot.CanvasAxisTickRenderer,
				    label: this.options.xaxis_label,
				    labelOptions: {fontSize:"10pt",
						   textColor:"black"},
				    labelRenderer:$.jqplot.CanvasAxisLabelRenderer,
				    min:this.options.mindate,
				    max:this.options.maxdate},
			    yaxis: {tickOptions:{formatString:"%0.2f",
						 fontSize:"8pt"},
				    tickRenderer:$.jqplot.CanvasAxisTickRenderer,
				    label: this.options.yaxis_label,
				    labelOptions: {fontSize:"10pt",
						   textColor:"black"},
				    labelRenderer:$.jqplot.CanvasAxisLabelRenderer}
	                  },
		    cursor: {show: true,
			     zoom: true,
			     constrainZoomTo:'x',
			     constrainOutsideZoom:false,
			     looseZoom: false,
			     showTooltip: false,
			     useAxesFormatters:false},
		    grid: {background:this.options.bg_color,
			   drawBorder:false,
			   shadow: false},
		    highlighter: {show:true,
				  fadeTooltip:false,
				  markerRenderer:new $.jqplot.MarkerRenderer(
                                                           {style:"circle",
							    shadow:false}),
				  sizeAdjust:20,
				  tooltipAxes:"z",
				  tooltipLocation:"se",
		                  bringSeriesToFront:true},
		    legend: legend_options,
		    series: series_options,
		    seriesColors: this.options.series_colors,
		    seriesDefaults: {pointLabels:{show:false}}
                  };

    // Create the plot
    this.plot = $.jqplot(this.id, data, this.config);

    // Store a data dictionary in the plot
    this.plot.data_dict = {};

    if (this.options.series_labels.length>1) {
	var tp = this;

	// Add click handler, to track selected series if the legend is clicked
	$("#"+tp.id).on("click","td.jqplot-table-legend",function(){
	    for (var series_i in tp.plot.series) {
		var label = tp.plot.series[series_i].label;
		var hidden = tp.plot.series[series_i]
		               .canvas._elem.hasClass('jqplot-series-hidden');
		tp.selected[label] = !hidden;
	    }
	});

	// Add hook, to be called after each draw:
	// If series not selected before the redraw, hide it
	var postDraw = function() {
	    $("#"+tp.id+" td.jqplot-table-legend-label").each(function(){
		var srs = $(this).text();
	        if (tp.selected[srs]!=undefined && !tp.selected[srs]) {
		    tp.plot.series[tp.series_index[srs]].toggleDisplay({data:0});
		    $(this).addClass('jqplot-series-hidden');
		    $(this).prev('.jqplot-table-legend-swatch')
		           .addClass('jqplot-series-hidden');
	        }
	    }); // end each legend label td
	};
	tp.plot.postDrawHooks.add(postDraw);
    }
}; // end init

TimePlot.prototype.composeHTML = function() {
    return '<div id='+this.id+' class="time_plot"></div>';
}; // end composeHTML


TimePlot.prototype.addRecord = function(records) {

    // If there is an existing plot, get its data and information
    // about which series are currently selected
    var data_dict = {};
    if (this.plot) {
	data_dict = this.plot.data_dict;
	for (var series_i in this.plot.series) {
	    var label = this.plot.series[series_i].label;
	    var hidden = this.plot.series[series_i]
		             .canvas._elem.hasClass('jqplot-series-hidden');
	    this.selected[label] = !hidden;
	}
    }

    // record: eg. {series:'V', date:'...', data:0.95, error:0.05}
    if (records) {
	if (!(records instanceof Array)) {
	    records = [records];
	}
	for (var i in records) {
	    var record = records[i];

	    if (data_dict[record['series']]==undefined) {
		data_dict[record['series']] = {};
	    }
	    data_dict[record['series']][record['key']] = record;
	}
    }

    var data = [], band_data = [];
    var i, j, series, recs, date, key,
        value, error, lower, upper, sdata, ldata, udata, dates;
    var series_to_use = this.options.series_labels;
    for (j in series_to_use) {
	series = series_to_use[j];

	if (!data_dict[series]) {
	    data.push([null]);	
	    band_data.push([[null],[null]]);
	} else {

	    sdata=[], ldata=[], udata=[], recs =[];

	    // Pull out dates first and sort them
	    ////here -- better way to do this?
	    for (key in data_dict[series]) {
	        recs.push(data_dict[series][key]);
	    }
	    recs.sort(function(a,b){
	        if (a.date<b.date) {
		    return -1;
		}
		if (a.date>b.date) {
		    return 1;
		}
		return 0;
	    });

	    for (i in recs) {
	        date = recs[i]['date'];
	        value = recs[i]['data'];
	        error = recs[i]['error'];
	        key = recs[i]['key'];

	        lower = value - error;
	        upper = value + error;
		
		// convert date to timestamp, and UT, if desired
		// This uses jqPlot's own date handling methods,
		// built on top of JavaScript's native Date object
		// There doesn't seem to be any way to truly
		// work in a non-local timezone, so hack it by
		// adding the UTC offset to the reported time
		date = new $.jsDate(date);
		if (this.options.ut) {
		    date = date.add(this.ut_offset,"milliseconds");
		}
		date = date.getTime();

	        sdata.push([date,value,key]);
	        ldata.push([date,lower]);
	        udata.push([date,upper]);
	    }
	    data.push(sdata);
	    band_data.push([ldata,udata]);
	}
    }

    // Check to see if min/maxdate should be in UT
    // and update it in options and config object
    if (this.options.ut && !this.ut) {
	this.options.mindate = new $.jsDate(this.options.mindate)
	                           .add(this.ut_offset,"milliseconds")
	                           .getTime();
	this.options.maxdate = new $.jsDate(this.options.maxdate)
		                   .add(this.ut_offset,"milliseconds")
		                   .getTime();
	var label_list = this.options.xaxis_label.split(" ");
	this.options.xaxis_label = label_list.slice(0,-1) + " (UT)";

    } else if (!this.options.ut && this.ut) {
	this.options.mindate = new $.jsDate(this.options.mindate)
		                   .add(-this.ut_offset,"milliseconds")
		                   .getTime();
	this.options.maxdate = new $.jsDate(this.options.maxdate)
		                   .add(-this.ut_offset,"milliseconds")
		                   .getTime();
	var label_list = this.options.xaxis_label.split(" ");
	this.options.xaxis_label = label_list.slice(0,-1)+
	                           " ("+this.timezone+")";
    }
    this.config.axes.xaxis.min = this.options.mindate;
    this.config.axes.xaxis.max = this.options.maxdate;
    this.config.axes.xaxis.label = this.options.xaxis_label;    

    // Update or create the plot
    if (this.plot) {
	this.plot.data = data;
	for (var series_i in data) {
	    var this_series = this.plot.series[series_i];
	    if (data[series_i][0]!=null) {
		this_series.data = data[series_i];

		var renderer = this_series.renderer;
		renderer.bands.show = true; 
		renderer.options.bandData = band_data[series_i];
		renderer.initBands.call(this_series, 
					renderer.options, this.plot);

		this.plot.series[series_i].showLabel = true;
	    } else {
		this.plot.series[series_i].data = [];
		this.plot.series[series_i].showLabel = false;
	    }
	}	

	// Check whether date should be displayed in UT, and it is not already
	var mindate, maxdate;
	if (this.options.ut && !this.ut) {
	    mindate = new $.jsDate(this.plot.axes.xaxis.min);
	    mindate.add(this.ut_offset,"milliseconds");
	    this.plot.axes.xaxis.min = mindate.getTime();

	    maxdate = new $.jsDate(this.plot.axes.xaxis.max);
	    maxdate.add(this.ut_offset,"milliseconds");
	    this.plot.axes.xaxis.max = maxdate.getTime();

	    this.plot.axes.xaxis._options.min = this.options.mindate;
	    this.plot.axes.xaxis._options.max = this.options.maxdate;
	    this.plot.axes.xaxis.labelOptions.label = this.options.xaxis_label;
	    this.plot.axes.xaxis.resetScale({min:mindate,max:maxdate});
	    this.ut = true;
	} else if (!this.options.ut && this.ut) {
	    mindate = new $.jsDate(this.plot.axes.xaxis.min);
	    mindate.add(-this.ut_offset,"milliseconds");
	    this.plot.axes.xaxis.min = mindate.getTime();

	    maxdate = new $.jsDate(this.plot.axes.xaxis.max);
	    maxdate.add(-this.ut_offset,"milliseconds");
	    this.plot.axes.xaxis.max = maxdate.getTime();

	    this.plot.axes.xaxis._options.min = this.options.mindate;
	    this.plot.axes.xaxis._options.max = this.options.maxdate;
	    this.plot.axes.xaxis.labelOptions.label = this.options.xaxis_label;
	    this.plot.axes.xaxis.resetScale({min:mindate,max:maxdate});
	    this.ut = false;
	}
	this.plot.replot({resetAxes:['yaxis']});
    } else {
	this.plot = $.jqplot(this.id, data, this.config);
    }

    // Store information about the plotted points in the data dictionary
    for (var i=0; i<series_to_use.length; i++) {
	s = this.plot.series[i];
	for (var j=0; j<s.gridData.length; j++) {
	    p = s.gridData[j];
	    var point = {seriesIndex:i, pointIndex:j, 
			 gridData:p, data:s.data[j]};
	    data_dict[s.label][s.data[j][2]]["point"] = point;
	}
    }

    // Store the data dictionary in the plot
    this.plot.data_dict = data_dict;

}; // end addRecord

TimePlot.prototype.highlightPoint = function(key) {
    // This function will highlight a data point given its key value

    if (!this.plot) {
	return;
    }

    // The highlight function is the mouseover event handler from
    // the Highlighter plugin
    var highlight_fn = function(plot, point) {
	$.jqplot.Highlighter.handleMove(null,null,null,point,plot);
    }

    // If no key, clear any existing highlights
    if (!key) {
	highlight_fn(this.plot);
	return;
    }

    // Get the record associated with the key
    var dd = this.plot.data_dict;
    var rec;
    for (var s in dd) {
	if (dd[s][key]) {
	    rec = dd[s][key];
	    break;
	}
    }

    // If no record found, clear existing highlights
    if (!rec) {
	highlight_fn(this.plot);
	return;
    }
    
    // Get the point information from the record
    var point = rec["point"];

    // Highlight the point
    highlight_fn(this.plot, point);

    return;
}

TimePlot.prototype.clearRecord = function() {
    // Clear out old plot
    if (this.plot) {
	this.plot.destroy();
	this.plot = null;
    }
    // Replace with an empty plot
    this.addRecord();

}; // end clearRecord


// TooltipOverlay child class
// constructor
// element should be a jQuery selection of an empty DOM element (ie. div),
// id is a string, and columns is an object containing column model information
function TooltipOverlay(element,id,selection) {
    this.element = element;
    this.id = id;
    this.selection = selection;
    this.messages = {};
    this.init();
}
// prototype: inherit from ViewPort
var to_proto = new ViewPort();
TooltipOverlay.prototype = to_proto;
TooltipOverlay.prototype.constructor = TooltipOverlay;

// add TooltipOverlay-specific methods
TooltipOverlay.prototype.init = function() {

    // Add an empty div to the element
    var html_str = this.composeHTML();
    this.element.append(html_str);

    // Add mouseenter/mouseleave event handlers to the selection
    var delay = 500;
    var messages = this.messages;
    var tooltip = $("#"+this.id);
    var selection = this.selection;
    $(document).on("mouseenter",selection,function() {

	var trigger = $(this);

	// Calculate position of tooltip
	var tt_left, tt_top;
	var trg_pos = trigger.offset();
	var trg_h =  trigger.outerHeight();
	var trg_w =  trigger.outerWidth();
	var tt_h = tooltip.outerHeight();
	var tt_w = tooltip.outerWidth();
	var screen_w = $(window).width();
	var screen_h = $(window).height();
	var overflow_btm = (trg_pos.top + trg_h + tt_h) - screen_h;
	if (overflow_btm>0) {
	    tt_top = trg_pos.top - tt_h - 5;
	} else {
	    tt_top = trg_pos.top + trg_h + 5;
	}
	var trg_ctr = trg_pos.left + Math.round(trg_w/2);
	var overflow_rt = (trg_ctr + tt_w) - screen_w;
	if (overflow_rt>0) {
	    tt_left = trg_ctr - overflow_rt - 5;
	} else {
	    tt_left = trg_ctr;
	}
	
	// Position tooltip
	tooltip.css({
	    left: tt_left,
	    top: tt_top,
	    position: "absolute"
	});

	// Add the message
	var key = trigger.attr("id");
	if (key==undefined) {
	    key = trigger.parent().attr("id");
	}
	var msg = messages[key];
	if (msg==undefined) {
	    msg = "No message";
	}
	tooltip.text(msg);

	// Set delay so that tooltip does not trigger on 
	// accidental mouseover
	hovering = setTimeout(function() {
	    // Show the tooltip
	    tooltip.show();
	    
	    // Throw a custom event, as a hook for the manager
	    // to do something (eg. hide any other tooltips)
	    tooltip.trigger("showTooltip");

	}, delay); // end timeout
	tooltip.data("hovering",hovering);
    }); // end mouseenter
    var tto = this;
    $(document).on("mouseleave",selection,function() {
        tto.clearRecord();
    }); // end mouseleave


}; // end init
TooltipOverlay.prototype.composeHTML = function() {
    return '<div id='+this.id+' class="tooltip" style="display:none"></div>';
}; // end composeHTML

TooltipOverlay.prototype.addRecord = function(records) {
    if (!(records instanceof Array)) {
	    records = [records];
    }
    for (var i in records) {
	var record = records[i];
	this.messages[record.key] = record.message;
    }
}; // end addRecord

TooltipOverlay.prototype.clearRecord = function() {
    var tooltip = $('#'+this.id);

    // Clear any current hovering states
    if (tooltip.data("hovering")) {
	clearTimeout(tooltip.data("hovering"));
	tooltip.data("hovering",null);
    }

    // Clear out message
    tooltip.text("");

    // Hide the tooltip
    tooltip.hide();

};
