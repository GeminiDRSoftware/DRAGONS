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
    addRecord: function(records,prepend) {
    	if (!(records instanceof Array)) {
    	    records = [records];
    	}
    	var record_str = "";
    	for (var i=0;i<records.length;i++) {
    	    if (prepend) {
    		record_str += records[records.length-i-1];
    	    } else {
    		record_str += records[i];
    	    }
    	}
    	if (prepend) {
    	    $('#'+this.id).prepend(record_str);
    	} else {
    	    $('#'+this.id).append(record_str);
    	}
        },
    clearRecord: function() {
    	$('#'+this.id).html("");
    }
};

// KeyedViewPort: this child class just overrides the addRecord method to
// accept a key, so that there is only one message with the given key
function KeyedViewPort(element,id,columns) {
    this.element = element;
    this.id = id;
    this.init();
}
// prototype: inherit from ViewPort
var kvp_proto = new ViewPort();
KeyedViewPort.prototype = kvp_proto;
KeyedViewPort.prototype.constructor = KeyedViewPort;

// add KeyedViewPort-specific methods
KeyedViewPort.prototype.addRecord = function(records,prepend) {
    if (!(records instanceof Array)) {
	records = [records];
    }
    var record_str = "";
    var record;
    for (var i=0;i<records.length;i++) {
	if (prepend) {
	    record = records[records.length-i-1];
	} else {
	    record = records[i];
	}
	var key = record["key"];
	// If a message with this key already exists, remove it
	if ($("#"+key).length>0) {
	    $("#"+key).remove();
	}
	record_str += '<span id="'+key+'">'+record["message"]+'</span>';
    }
    if (prepend) {
	$('#'+this.id).prepend(record_str);
    } else {
	$('#'+this.id).append(record_str);
    }

}; // end addRecord


// SimpleTable child class
// constructor
// element should be a jQuery selection of an empty DOM element (ie. div),
// id is a string, and columns is an object containing column model information
function SimpleTable(element,id,columns) {
    this.element = element;
    this.id = id;
    this.columns = columns;
    this.rows = {};
    this.records = {};
    this.init();
}
// prototype: inherit from ViewPort
var st_proto = new ViewPort();
SimpleTable.prototype = st_proto;
SimpleTable.prototype.constructor = SimpleTable;

// add SimpleTable-specific methods
SimpleTable.prototype.composeHTML = function() {
    var html_str = "";

    // Compose table tags with column headers
    html_str += '<table class="simple_table" id="'+this.id+'"><thead><tr>';
    for (i in this.columns) {
	col = this.columns[i];

	// Add classes for column id, as well swap, hidden, and searchable
	// properties
	html_str += '<th  class="'+col.id+'">'+col.name+'</th>';
    }
    html_str += '</tr></thead>';

    // Add table body
    html_str += '<tbody></tbody>';

    // Add table footer
    html_str += '<tfoot></tfoot>';

    html_str += '</table>';

    return html_str;
}; // end composeHTML

SimpleTable.prototype.addRecord = function(records) {
    // Make records into an array if it is not already
    if (!records) {
	records = [];
    }
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

	for (var j in this.columns) {
	    col = this.columns[j];
	    table_row += '<td class="'+col.id+'">'+record[col.field]+'</td>';
	}
	table_row += '</tr>';

	// Add the new row to the database of rows
	if ($("#"+record['key']).length>0) {
	    // A row with this key exists already, remove it
	    $("#"+record['key']).remove();
	}

	// Append the row to the table body
	tbody.append(table_row);

	// Update the rows database with this row
	this.rows[record['key']] = table_row;
    }

    // Add even classes to all rows to allow them to be styled
    $("#"+this.id+" tbody tr").removeClass("even");
    $("#"+this.id+" tbody tr:even").addClass("even");

}; // end addRecord

SimpleTable.prototype.clearRecord = function() {

    // Clear out the tbody
    $('#'+this.id+' tbody').html("");

    // Reset records and rows objects
    this.records = {};
    this.rows = {};
}; // end clearRecord


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

    // Add the table html to the element
    var html_str = this.composeHTML();
    this.element.html(html_str);

    // Hide any hidden columns
    $("#"+this.id+" td.hidden,th.hidden").hide();

    // Keep a deep copy of the initial columns to revert to on clearRecord
    this._columns = [];
    for (var i=0;i<this.columns.length;i++) {
	var c = $.extend(true,{},this.columns[i]);
	this._columns.push(c);
    }

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

    }); // end on click th.swap

    // Add event handlers to the search field to allow row filtering
    this.element.on("focus","input.filter",function() {
	if ($(this).hasClass("no_input")) {
	    $(this).attr("value","");
	}
	$(this).removeClass("no_input");
	return false;
    }); // end on focus
    this.element.on("blur","input.filter",function() {
	if ($(this).hasClass("no_input") || $(this).attr("value")=="") {
	    $(this).addClass("no_input");
	    $(this).attr("value","Search");
	}
	return false;
    }); // end on blur
    this.element.on("keyup","input.filter",function() {
	$(this).removeClass("no_input");
	var query = $(this).attr("value");
	st.filterRecord(query);
	return false;
    }); // end on keypress

    // Make the column headers draggable
    $("#"+this.id+" th").not(".hidden,.pad").draggable({
        cursor: "move",
	helper: "clone",
	revert: true,
	revertDuration: 200,
	zIndex: 100,
	start: function(ev,ui) {
		var drg_id;
		for (var col_i=0;col_i<st.columns.length;col_i++) {
		    var col = st.columns[col_i];
		    if ($(this).hasClass(col.id)) {
			drg_id = col.id;
			break;
		    }
		}
		ui.helper.data("col_id",drg_id);
		$("#"+st.id+" td."+drg_id).addClass("ui-draggable-dragging");
	    },
	stop: function(ev,ui) {
		var drg_id = ui.helper.data("col_id");
		$("#"+st.id+" td."+drg_id).removeClass("ui-draggable-dragging");
	    }
    }); // end draggable

    // Make them droppable too
    $("#"+this.id+" th").not(".hidden,.pad").droppable({
	accept: "#"+this.id+" th",
        drop: reorderColumns,
	hoverClass: "hover",
	tolerance: "pointer"
    }); // end draggable

    // Callback for drop event
    function reorderColumns(ev,ui) {
	var dragged = ui.draggable;
	var target = $(this);
	var tgt_i, drg_i;
	var last_col = st.columns[st.columns.length-1];
	for (var col_i=0;col_i<st.columns.length;col_i++) {
	    var col = st.columns[col_i];
	    if (target.hasClass(col.id)) {
		tgt_i = col_i;
	    }
	    if (dragged.hasClass(col.id)) {
		drg_i = col_i;
	    }
	}
	if (tgt_i!=drg_i) {
	    // Stop header from reverting to original position
	    dragged.draggable("option","revert",false);

	    // Change the order in the column model
	    var drg_col = st.columns[drg_i];
	    st.columns.splice(drg_i,1);
	    st.columns.splice(tgt_i,0,drg_col);

	    if (tgt_i<drg_i) {

		// Move the header cell
		$("#"+st.id+" th."+drg_col.id)
		    .insertBefore($("#"+st.id+" th").eq(tgt_i));

		// Move the data cells
		$("#"+st.id+" tbody td."+drg_col.id).each(function(){
		    var target = $(this).parent().find("td").eq(tgt_i);
		    $(this).insertBefore(target);
		}); // end each cell

	    } else {
		// Do the same, but insertAfter instead of before
		$("#"+st.id+" th."+drg_col.id)
		    .insertAfter($("#"+st.id+" th").eq(tgt_i));
		$("#"+st.id+" tbody td."+drg_col.id).each(function(){
		    var target = $(this).parent().find("td").eq(tgt_i);
		    $(this).insertAfter(target);
		}); // end each cell
	    }
	}
    }

}; // end init

ScrollTable.prototype.composeHTML = function() {
    var html_str = "";

    // Compose table tags with column headers
    html_str += '<table class="scroll_table" id="'
                +this.id+'" style="table-layout:fixed ">' +
                '<thead class="fixed_header">';
    html_str += '<tr style="position:relative;display:block">'
    for (i in this.columns) {
	col = this.columns[i];

	// Add classes for column id, as well swap, hidden, and searchable
	// properties
	html_str += '<th  class="'+col.id;
	var w = col.width;
	if (col.swap) {
	    html_str += ' swap';
	}
	if (col.hidden) {
	    w = 0;
	    html_str += ' hidden';
	}
	if (!col.disable_search) {
	    html_str += ' searchable';
	}
	html_str += '"';

	// Set column width
	html_str += ' style="width:'+w+'px"';
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
    html_str += '<tbody class="scroll_body"'+
	        'style="display:block;overflow:auto;' +
                'position:relative;width:100%;height:'+
                (parseInt(this.element.height(),10)-32-32) +'px' +
                '">';
    html_str += '</tbody>';

    // Add table footer
    html_str += '<tfoot class="fixed_footer">';
    html_str += '<tr style="display:block;position:relative;">' +
                '<td class="filter_message">Displaying 0 of 0 rows</td>'+
                '<td class="filter">' +
                '<div style="position:relative">' +
                '<input class="filter no_input" type=text value="Search">' +
                '<span class="search_icon"></span></div></td>'+
                '</tr>';
    html_str += '</tfoot>';


    html_str += '</table>';

    return html_str;
}; // end composeHTML

ScrollTable.prototype.addRecord = function(records) {
    // Make records into an array if it is not already
    if (!records) {
	records = [];
    }
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

	    var w = col.width;
	    var display_str = "";
	    table_row += '<td class="'+col.id;
	    if (col.hidden) {
		w = 0;
		display_str = " display:none";
		table_row += ' hidden';
	    }
	    if (!col.disable_search) {
		table_row += ' searchable';
	    }
	    table_row += '" style="width:'+w+'px;'+display_str+'">'+
		         record[col.field]+'</td>';
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

    // Hide any hidden columns
    $("#"+this.id+" td.hidden,th.hidden").hide();

    // Check for overflow: if present, add a 16px pad to the table header
    $("#"+this.id+" th.pad").remove();
    if (tbody[0].clientHeight!=tbody[0].scrollHeight) {
	$("#"+this.id+" thead tr").append(
	    '<th class="pad" style="width:16px;padding:0px"></th>');
    }

    // Filter records if needed
    $("#"+this.id+" tbody tr").addClass("visible");
    var query = "";
    if (!$("#"+this.id+" input.filter").hasClass("no_input")) {
	query = $("#"+this.id+" input.filter").attr("value");
    }
    this.filterRecord(query);

    // Add even classes to all rows to allow them to be styled
    $("#"+this.id+" tbody tr").removeClass("even");
    $("#"+this.id+" tbody tr.visible:even").addClass("even");

}; // end addRecord

ScrollTable.prototype.clearRecord = function() {

    // Restore initial column configuration
    this.columns = this._columns;

    // Make a new deep copy of the original configuration
    // and reset the thead
    this._columns = [];
    var thead_tr = $('#'+this.id+' thead tr');
    var thead_th = $('#'+this.id+' thead th').detach();
    thead_tr.html("");
    for (var i=0;i<this.columns.length;i++) {
	var col = this.columns[i];
	var c = $.extend(true,{},col);
	this._columns.push(c);

	thead_tr.append(thead_th.filter("."+col.id));
    }
    thead_tr.append(thead_th.filter(".pad"));

    // Clear out the tbody
    $('#'+this.id+' tbody').html("");

    // Clear out the filtering box in tfoot
    $("#"+this.id+" input.filter").addClass("no_input").attr("value","Search");
    this.filterRecord();

    // Reset records and rows objects
    this.records = {};
    this.rows = {};
}; // end clearRecord

ScrollTable.prototype.filterRecord = function(query) {
    // Trim out leading/trailing whitespace
    query = $.trim(query);

    if (query=="") {
	$("#"+this.id+" tbody tr")
	    .removeClass("even")
	    .addClass("visible")
	    .show();
    } else {
	// Replace any remaining whitespace with OR
	query = query.replace(/ /gi, '|');

	// Escape any \
	query = query.replace(/(\\)/gi, '\\\\');

	// Get non-searchable column names
	var no_search = [];
	for (var col in this.columns) {
	    col = this.columns[col];
	    if (col.disable_search) {
		no_search.push(col.id);
	    }
	}

	var regex_query = new RegExp(query, "i");

	$("#"+this.id+" tbody tr").each(function(){
	    if ($(this).find("td.searchable").text().search(regex_query)<0) {
		$(this).removeClass("visible").hide();
	    } else {
		$(this).addClass("visible").show();
	    }
	    $(this).removeClass("even");
	}); // end each tr
    }

    // Redo even row classes
    $("#"+this.id+" tbody tr.visible:even").addClass("even");

    // Display the number of visible rows
    var display_msg = "Displaying " +
                      $("#"+this.id+" tbody tr.visible").length +
                      " of " +
                      $("#"+this.id+" tbody tr").length +
                      " rows";
    $("#"+this.id+" tfoot td.filter_message").text(display_msg);

}; // end filterRecord


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

    // Check whether date should be displayed in UT or LT
    var date = new $.jsDate;
    if (this.options.ut_offset) {
	this.ut_offset = this.options.ut_offset * 3600000;
    } else {
	this.ut_offset = date.getUtcOffset();
    }
    if (this.options.timezone) {
	this.timezone = this.options.timezone;
    } else {
	this.timezone = date.getTimezoneAbbr();
    }
    var mindate, maxdate;
    var mindate = new $.jsDate(this.options.mindate);
    var maxdate = new $.jsDate(this.options.maxdate);
    if (this.options.ut && !this.ut) {
	mindate.add(this.ut_offset,"milliseconds");
	maxdate.add(this.ut_offset,"milliseconds");
	this.ut = true;
    } else if (!this.options.ut && this.ut) {
	mindate.add(-this.ut_offset,"milliseconds");
	maxdate.add(-this.ut_offset,"milliseconds");
	this.ut = false;
    }

    this.options.mindate = mindate.getTime();
    this.options.maxdate = maxdate.getTime();
    // Prevent multiple date labels being added to the x-axis on re-plotting
    if (this.options.xaxis_label.substr(this.options.xaxis_label.length - 1, 1) != ')') {
        if (this.ut) {
	    this.options.xaxis_label += " (UT)";
        } else {
	    this.options.xaxis_label += " ("+this.timezone+")";
        }
    }

    // Check whether y-axis should be inverted
    var ymin, ymax;
    if (this.options.invert_yaxis) {
	ymin = this.options.ymax;
	ymax = this.options.ymin;
    } else {
	ymin = this.options.ymin;
	ymax = this.options.ymax;
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
						   textColor:"DarkSeaGreen"},
				    labelRenderer:$.jqplot.CanvasAxisLabelRenderer,
				    min: this.options.mindate,
				    max: this.options.maxdate},
			    yaxis: {min: ymin,
				    max: ymax,
		                    tickOptions:{formatString:"%0.2f",
						 fontSize:"8pt"},
				    tickRenderer:$.jqplot.CanvasAxisTickRenderer,
				    label: this.options.yaxis_label,
				    labelOptions: {fontSize:"10pt",
						   textColor:"DarkSeaGreen"},
				    labelRenderer:$.jqplot.CanvasAxisLabelRenderer}
	                  },
		    canvasOverlay: {show:false,objects:[]},
		    cursor: {show: true,
			     zoom: true,
			     constrainOutsideZoom:false,
			     looseZoom: true,
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
                  };

    // Create the plot
    this.plot = $.jqplot(this.id, data, this.config);

    // Store a data dictionary in the plot
    this.plot.data_dict = {};

    var tp = this;
    if (this.options.series_labels.length>1) {
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
	var hideUnselected = function() {
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
	tp.plot.postDrawHooks.add(hideUnselected);
    }

    // Add hook, to be called after each draw:
    // Update point information in the data dictionary
    var updatePoints = function() {
	for (var i=0; i<tp.options.series_labels.length; i++) {
	    var s = tp.plot.series[i];
	    for (var j=0; j<s.gridData.length; j++) {
		p = s.gridData[j];
		var point = {seriesIndex:i, pointIndex:j,
			     gridData:p, data:s.data[j]};
		tp.plot.data_dict[s.label][s.data[j][2]]["point"] = point;
	    }
	}
    };
    tp.plot.postDrawHooks.add(updatePoints);

    // Add handler for jqplotDataPointHighlight event to pull up
    // conditions lines
    tp.element.on("jqplotDataPointHighlight","div.jqplot-target",function(ev,pt){
        if (tp.options.overlay.length<1) {
	    return;
	}

	var co = tp.plot.plugins.canvasOverlay;
	co.options.show=true;

	var series_index = undefined;
	for (var i=0; i<tp.options.series_labels.length; i++) {
	    var s = tp.options.series_labels[i];
	    if (tp.plot.data_dict[s] &&
		tp.plot.data_dict[s][pt.data[2]]!=undefined) {
		series_index = i;
		break;
	    }
	}
	if (series_index==undefined) {
	    return;
	}
	objects = tp.options.overlay[series_index];
	for (var line in objects) {
	    line = objects[line];
	    var object = {name:line.name,
			  show:true,
			  showLabel:true,
			  labelLocation:'e',
			  labelOffset:8,
			  y:line.y,
			  color:line.color,
			  shadow:false,
			  dashPattern:[2,3],
			  lineWidth:1};
	    if ($.inArray(object.name,co.objectNames)==-1) {
		co.addDashedHorizontalLine(object);
	    }
	}
	co.draw(tp.plot);
    }); // end on highlight
    tp.element.on("jqplotDataPointUnhighlight","div.jqplot-target",function(ev){
	var co = tp.plot.plugins.canvasOverlay;
	co.options.show = false;
	co.clear();
	co.objects = [];
	co.objectNames = [];
    }); // end on unhighlight

    // Update the HTML to ensure the title which contains the optical/IR switch is accessible
    tp.plot.postDrawHooks.add(function() {
        $('#' + tp.id + ' .jqplot-title').remove().appendTo('#' + tp.id);
    });

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

    var data = [], band_data = [], y_values = [];
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

	    // Sort the records by date
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
		y_values.push(lower, upper);
	    }
	    data.push(sdata);
	    band_data.push([ldata,udata]);
	}
    }

    // Get new min/max for y-axis
    // Invert them if desired
    if (y_values.length>0) {

	// Get ymin/ymax values from data
	var ymin, ymax;
	if (!this.options.invert_yaxis) {
	    ymin = Math.min.apply(null,y_values);
	    ymax = Math.max.apply(null,y_values);

	    // Add 20% padding, round to nearest 0.1
	    var range = Math.abs(ymax-ymin)*1.2;
	    var ctr = (ymax+ymin)/2;
	    ymin = Math.floor((ctr - range/2)*10)/10;
	    ymax = Math.ceil((ctr + range/2)*10)/10;

	    // Check to see if an absolute max/min was defined in options
	    if ((this.options.ymin!=undefined && ymin<=this.options.ymin) ||
		(this.options.ymax!=undefined && ymin>=this.options.ymax))
	    {
		ymin = this.options.ymin;
	    }
	    if ((this.options.ymax!=undefined && ymax>=this.options.ymax) ||
		(this.options.ymin!=undefined && ymax<=this.options.ymin))
	    {
		ymax = this.options.ymax;
	    }
	} else {

	    // Invert min and max
	    ymin = Math.max.apply(null,y_values);
	    ymax = Math.min.apply(null,y_values);

	    var range = Math.abs(ymax-ymin)*1.2;
	    var ctr = (ymax+ymin)/2;
	    ymax = Math.floor((ctr - range/2)*10)/10;
	    ymin = Math.ceil((ctr + range/2)*10)/10;

	    if ((this.options.ymin!=undefined && ymax<=this.options.ymin) ||
		(this.options.ymax!=undefined && ymax>=this.options.ymax))
            {
		ymax = this.options.ymin;
	    }
	    if ((this.options.ymax!=undefined && ymin>=this.options.ymax) ||
		(this.options.ymin!=undefined && ymin<=this.options.ymin))
	    {
		ymin = this.options.ymax;
	    }
	}

	// Set min/max in plot
	if (this.plot.axes.yaxis._options.min==undefined ||
	    this.plot.axes.yaxis._options.max==undefined ||
	    (this.plot.axes.yaxis._options.min==this.plot.axes.yaxis.min &&
	     this.plot.axes.yaxis._options.max==this.plot.axes.yaxis.max))
	{
	    this.plot.axes.yaxis.min = ymin;
	    this.plot.axes.yaxis.max = ymax;
	    this.plot.axes.yaxis._options.min = ymin;
	    this.plot.axes.yaxis._options.max = ymax;
	    this.plot.axes.yaxis.resetScale({min:ymin,max:ymax});
	} else {
	    this.plot.axes.yaxis._options.min = ymin;
	    this.plot.axes.yaxis._options.max = ymax;
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
	this.options.xaxis_label = label_list.slice(0,-1) +
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
	    this.plot.axes.xaxis.resetScale({min:mindate,
			                     max:maxdate});
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
	    this.plot.axes.xaxis.resetScale({min:mindate,
			                     max:maxdate});
	    this.ut = false;
	}
	this.plot.replot({resetAxes:false});
    } else {
	this.plot = $.jqplot(this.id, data, this.config);

	// Add hooks, to be called after each draw:
	// (The rest of the event handlers established in init
	// will still work because they are jQuery 'on' handlers
	var tp = this;

	// If series not selected before the redraw, hide it
	if (this.options.series_labels.length>1) {
	    var hideUnselected = function() {
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
	    tp.plot.postDrawHooks.add(hideUnselected);
	}
	// Update point information in the data dictionary
	var updatePoints = function() {
	    for (var i=0; i<tp.options.series_labels.length; i++) {
		var s = tp.plot.series[i];
		for (var j=0; j<s.gridData.length; j++) {
		    p = s.gridData[j];
		    var point = {seriesIndex:i, pointIndex:j,
				 gridData:p, data:s.data[j]};
		    tp.plot.data_dict[s.label][s.data[j][2]]["point"] = point;
		}
	    }
	};

        // Update the HTML to ensure the title which contains the optical/IR switch is accessible
        tp.plot.postDrawHooks.add(function() {
            $('#' + tp.id + ' .jqplot-title').remove().appendTo('#' + tp.id);
        });

	tp.plot.postDrawHooks.add(updatePoints);

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
	if (!point) {
	    point = null
        }
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

    // Check whether point is currently visible
    try {
	if (this.options.invert_yaxis) {
	    if (point.data[0]>this.plot.axes.xaxis.max ||
		point.data[0]<this.plot.axes.xaxis.min ||
		point.data[1]>this.plot.axes.yaxis.min ||
		point.data[1]<this.plot.axes.yaxis.max) {

		// If not, clear existing highlights
		highlight_fn(this.plot);
		return;
	    }
	} else {
	    if (point.data[0]>this.plot.axes.xaxis.max ||
		point.data[0]<this.plot.axes.xaxis.min ||
		point.data[1]>this.plot.axes.yaxis.max ||
		point.data[1]<this.plot.axes.yaxis.min) {

		// If not, clear existing highlights
		highlight_fn(this.plot);
		return;
	    }
	}
    } catch (e) {
	console.log("Cannot highlight point: currently not visible");
	return;
    }
    // Highlight the point
    highlight_fn(this.plot, point);

    return;
};

TimePlot.prototype.updateDate = function(mindate,maxdate,xaxis_label) {
    // Assume incoming dates are LT
    this.ut = false;

    // Convert to UT milliseconds
    mindate = new $.jsDate(mindate);
    maxdate = new $.jsDate(maxdate);
    mindate = mindate.getTime();
    maxdate = maxdate.getTime();

    // Keep it in the options
    this.options.mindate = mindate;
    this.options.maxdate = maxdate;
    if (xaxis_label) {
	xaxis_label += " ("+this.timezone+")";
	this.options.xaxis_label = xaxis_label;
    }

    // Keep it in the jqplot config
    this.config.axes.xaxis.min = mindate;
    this.config.axes.xaxis.max = maxdate;
    if (xaxis_label) {
	this.config.axes.xaxis.label = xaxis_label;
    }

    // Update it in the plot if it exists
    if (this.plot) {
	// Set the unzoomed min/max
	this.plot.axes.xaxis._options.min = mindate;
	this.plot.axes.xaxis._options.max = maxdate;

	// Set the actual min/max
	this.plot.axes.xaxis.min = mindate;
	this.plot.axes.xaxis.max = maxdate;

	// Reset the label
	if (xaxis_label) {
	    this.plot.axes.xaxis.labelOptions.label = xaxis_label;
	}
	// Reset the scale
	this.plot.axes.xaxis.resetScale({min:mindate,
					 max:maxdate});
    }

    // Update the plot
    this.addRecord();

}; // end updateDate

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

	// Add the message
	var key = trigger.attr("id");
	if (key==undefined) {
	    key = trigger.parent().attr("id");
	}
	var msg = messages[key];
	if (msg==undefined) {
	    // If no message defined, do not display tooltip
	    return;
	}
	tooltip.text(msg);

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
