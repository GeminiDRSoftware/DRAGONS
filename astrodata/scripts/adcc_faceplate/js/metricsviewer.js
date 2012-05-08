// Driver class to instantiate all the viewports and hold the database
// of records

// Constructor
function MetricsViewer(element, id) {
    this.element = element;
    this.id = id;

    // Placeholders for ViewPorts
    this.metrics_table = null;
    this.message_window = null;
    this.iq_plot = null;
    this.cc_plot = null;
    this.bg_plot = null;
    this.tooltips = null;

    // Placeholder for database of records
    this.database = null;

    // Placeholder for adcc command pump
    this.gjs = null;

    // Initialize the viewer
    this.init();
}
// Add methods to prototype
MetricsViewer.prototype = {
    constructor: MetricsViewer,
    init: function() {

	// Reference to self, to use in functions inside init
	var mv = this;

	// Instantiate the records database
	this.database = new MetricsDatabase();

	// Add the basic structure to the element
	var html_str = this.composeHTML();
	this.element.html(html_str);

	// Add a date to the date box
	this.date_str = this.getDateString();
	$("#date").html(this.date_str);

	// Instantiate scrolling table
	// Define table columns
	var columns = [
                       {id:"datetime", name:"Time", 
			field:"metadata-local_time", width:56,
			sort:true, hidden:true},
                       {id:"time", name:"LT",
			field:"metadata-local_time_str", width:56,
		        swap: "metadata-ut_time_str", alt_name:"UT"},
                       {id:"imgnum", name:"Img#",
			field:"metadata-image_number", width:54},
		       {id:"datalabel",	name:"Data Label", 
			field:"metadata-datalabel", width:180},
		       {id:"wlen", name:"Wlen",
			field:"metadata-wavelength_str", width:60},
		       {id:"iq", name:"IQ",
			field:"iq-band", width:50},
		       {id:"cc", name:"CC",
			field:"cc-band", width:50},
		       {id:"bg", name:"BG",
			field:"bg-band", width:50},
		       {id:"deliq", name:"Delivered IQ",
			field:"iq-delivered_str", width:100},
		       {id:"zeniq", name:"Zenith IQ",
			field:"iq-zenith_str",width:100},
		       {id:"zeropt", name:"Zeropoint",
			field:"cc-zeropoint_str",width:100},
		       {id:"extinc", name:"Extinction", 
			field:"cc-extinction_str",width:100},
		       {id:"sky", name:"Sky Brightness", 
			field:"bg-brightness_str",width:114}
		       ]; // end columns
	this.metrics_table = new ScrollTable($("#table_wrapper"),
					     "metrics_table",columns);

	// Instantiate message window
	this.message_window = new ViewPort($("#message_target"),"message_window");

	// Instantiate plots

	// Set up necessary for all plots
	var mindate = new Date();
	mindate.setHours(18);
	mindate.setMinutes(0);
	mindate.setSeconds(0);
	var maxdate = new Date();
	maxdate.setHours(mindate.getHours()+12);
	maxdate.setMinutes(0);
	maxdate.setSeconds(0);
	var options = {mindate: mindate,
		       maxdate: maxdate,
	               series_colors: ["#A0A0FF"],
	               bg_color: "white",
		       series_selectable: false,
	               title: "",
		       ////here -- need to fix label
	               xaxis_label: this.date_str,
	               yaxis_label: ""};
	
	// IQ Plot
	var iq_options = $.extend(true,{},options);
	iq_options.title = "Zenith IQ";
	iq_options.yaxis_label = "Zenith IQ (arcsec)";
	iq_options.series_selectable = true;
	iq_options.series_colors = ["red","orange","yellow","blue"];
	this.iq_plot = new TimePlot($("#iq_plot_wrapper"),"iqplot",iq_options);

	// CC Plot
	var cc_options = $.extend(true,{},options);
	cc_options.title = "Cloud Extinction";
	cc_options.yaxis_label = "Extinction (mag)";
	this.cc_plot = new TimePlot($("#cc_plot_wrapper"),"ccplot",cc_options);

	// BG Plot
	var bg_options = $.extend(true,{},options);
	bg_options.title = "Sky Brightness";
	bg_options.yaxis_label = "Sky Brightness (mag/arcsec^2)";
	this.bg_plot = new TimePlot($("#bg_plot_wrapper"),"bgplot",bg_options);


	// Instantiate tooltips
	this.tooltips = {};

	// Airmass, for zenith IQ table column
	var am_tt = new TooltipOverlay($("#tooltip_wrapper"),"tooltip_airmass",
				       "#metrics_table td.zeniq");
	this.tooltips["metadata-airmass"] = am_tt;

	// Requested metrics, for metrics band table columns
	var metrics = ["iq","cc","bg"];
	var i, tt, met;
	for (i in metrics) {
	    met = metrics[i];
	    tt = new TooltipOverlay($("#tooltip_wrapper"),"tooltip_"+met,
				    "#metrics_table td."+met);
	    this.tooltips[met+"-requested"] = tt;
	}

	// Full filename, for image number column
	var fn_tt = new TooltipOverlay($("#tooltip_wrapper"),"tooltip_filename",
				       "#metrics_table td.imgnum");
	this.tooltips["metadata-filename"] = fn_tt;

	// Obstype, for datalabel column
	var obs_tt = new TooltipOverlay($("#tooltip_wrapper"),"tooltip_obstype",
					"#metrics_table td.datalabel");
	this.tooltips["metadata-obstype"] = obs_tt;

	// Waveband, for wavelength column
	var wb_tt = new TooltipOverlay($("#tooltip_wrapper"),"tooltip_wband",
					"#metrics_table td.wlen");
	this.tooltips["metadata-waveband"] = wb_tt;

	// Add a hover class to table cells on mouseenter/mouseleave
	$("#metrics_table").on("mouseenter","td",function(){
		$(this).addClass("hover");
	    });
	$("#metrics_table").on("mouseleave","td",function(){
		$(this).removeClass("hover");
	    });

	// Add a handler for the showTooltip event: when a tooltip appears,
	// all others should hide
	$(document).on("showTooltip","div.tooltip",function(){
	    tt_shown = $(this);
	    for (var tt in mv.tooltips) {
		tt = mv.tooltips[tt];
		if (tt.id!=tt_shown.attr("id")) {
		    tt.clearRecord();
		}
	    }
	    // Stop event from bubbling up
	    return false;
	});
	// Also hide tooltips when mouse leaves table wrapper
	$(document).on("mouseleave","#table_wrapper",function(){
	    for (var tt in mv.tooltips) {
		tt = mv.tooltips[tt];
		tt.clearRecord();
	    }
	    // Stop event from bubbling up
	    return false;		
	});

	// Add event handler to link message window to clicks in table rows
	$("#metrics_table tbody").on("click", "tr", function() {
	    var selected = $(this).hasClass("highlight");

	    // clear previous selections 
	    $("#metrics_table tbody tr").removeClass("highlight");
	    $("#metrics_table tbody tr:even").addClass("even");
	    mv.message_window.clearRecord();
	
	    // add new selection (if not deselecting)
	    if (!selected) {
		$(this).removeClass("even");
		$(this).addClass("highlight");
		var dl = $(this).find("td.datalabel").text();
		var record = mv.database.getRecord(dl);
		var msg = mv.formatMessageRecords(record,"comment");
	        mv.message_window.addRecord(msg);
	    }
	}); // end click

	// Set the previous and next turnover times
	// Turnover is set to 14:00 at both sites
	var prev_turnover = new Date();
	if (prev_turnover.getHours()<14) {
	    prev_turnover.setDate(prev_turnover.getDate()-1);
	}
	prev_turnover.setHours(14);
	prev_turnover.setMinutes(0);
	prev_turnover.setSeconds(0);
	
	var next_turnover = new Date(prev_turnover.getFullYear(),
				     prev_turnover.getMonth(),
				     prev_turnover.getDate() + 1,
				     prev_turnover.getHours());
	
	// Use previous turnover as initial timestamp (in UTC seconds)
	// for adcc query
	var timestamp = Math.round(prev_turnover.valueOf()/1000);

	// Store next turnover to check against in the update function
	this.turnover = next_turnover;

	// Set up a mouse-position tracker
	this.last_pos = {x:0,y:0};
	$(document).mousemove(function(e) {
		mv.last_pos = {x:e.pageX,y:e.pageY};
	});

	// Hook up the adcc command pump to the update function
	this.gjs = new GJSCommandPipe();
	this.gjs.registerCallback("stat",function(msg){mv.update(msg);});
	this.gjs.startPump(timestamp);

    }, // end init

    composeHTML: function() {
	var html_str = "";

	// Outer wrapper
	html_str += '<div id='+this.id+' class="metrics_viewer">';
	
	// Date prefix box
	html_str += '<div id="date_wrapper"><span class="label">'+
	            'Date prefix: </span><span id="date"></span></div>';

	// Table wrapper
	html_str += '<div id="table_wrapper"></div>';

	// Message window wrapper
	html_str += '<div id="message_wrapper"><p>' +
	            '<span class="label">Messages</span>' +
	            '<span id="message_target"></span>' +
	            '</p></div>';
	
	// Plot wrapper
	html_str += '<div id="plot_wrapper">'+
	            '<div id="iq_plot_wrapper"></div>'+
	            '<div id="cc_plot_wrapper"></div>'+
	            '<div id="bg_plot_wrapper"></div>'+
	            '</div>';

	// Tooltip wrapper
	html_str += '<div id="tooltip_wrapper"></div>';

	// End outer wrapper
	html_str += '</div>';

	return html_str;
    }, // end composeHTML

    getDateString: function() {
	var date_str = "";
	var date = new Date();
	
	// Timezone offset is in minutes.
	// 600 is Hawaii; assume anything else is Chile
	var timezone = date.getTimezoneOffset();
	if (timezone==600) {
	    // Hawaii; UT date will be correct at nighttime
	    date_str += "N";
	} else {
	    // Chile
	    date_str += "S";

	    // Add one day, since the UT date we use at night is 
	    // actually the UT date for the next day
	    date = new Date(date.getFullYear(),
			    date.getMonth(),date.getDate()+1);
	}

	var year = date.getUTCFullYear();

	// Pad month and day with 0 if necessary
	var month = date.getUTCMonth() + 1;
	month = (month <10 ? "0" : "") +  month;
	var day = date.getUTCDate();
	day = (day <10 ? "0" : "") +  day;

	date_str += year + month + day + "S";
	return date_str;
    },

    isHover: function(element) {
	// Return true if element (a jQuery selection) is moused-over

	var pos = this.last_pos;
	var offset = element.offset();
	var width = element.outerWidth();
	var height = element.outerHeight();
	var is_hover = offset.left<=pos.x && offset.left+width>pos.x &&
	               offset.top<=pos.y && offset.top+height>pos.y;

	return is_hover;
    },

    reset: function() {
	// Reset everything for date turnover

	// Stop the pump
	this.gjs.stopPump();

	// Clear the database
	this.database.clearDatabase();

	// Update date string
	this.date_str = this.getDateString();
	$("#date").html(this.date_str);

	// Clear all ViewPorts
	this.metrics_table.clearRecord();
	this.message_window.clearRecord();
	this.iq_plot.clearRecord();
	this.cc_plot.clearRecord();
	this.bg_plot.clearRecord();
	for (var tt in this.tooltips) {
	    this.tooltips[tt].clearRecord();
	}

	// Set the turnover times
	var prev_turnover = new Date();
	if (prev_turnover.getHours()<14) {
	    prev_turnover.setDate(prev_turnover.getDate()-1);
	}
	prev_turnover.setHours(14);
	prev_turnover.setMinutes(0);
	prev_turnover.setSeconds(0);
	var timestamp = Math.round(prev_turnover.valueOf()/1000);
	
	var next_turnover = new Date(prev_turnover.getFullYear(),
				     prev_turnover.getMonth(),
				     prev_turnover.getDate() + 1,
				     prev_turnover.getHours());
	this.turnover = next_turnover;

	// Restart the pump
	this.gjs.startPump(timestamp);
    },

    update: function(records) {

	// Check the time to see if the page needs to be turned over
	var current_time = new Date();
	if (current_time > this.turnover) {
	    this.reset();
	}

	// Test input; make into an array if needed
	if (records==undefined) {
	    return;
	}
	if (!(records instanceof Array)) {
	    records = [records];
	} else if (records.length==0) {
	    return;
	}

	for (var i in records) {
	    var record = records[i];

	    // Add a few more useful fields to the record

	    // Get just the hours and minutes from the local time
	    // Input format: "YYYY-MM-DD HH:MM:SS"
	    var time = record["metadata"]["local_time"];
	    time = time.split(" ")[1].split(":",2).join(":");
	    record["metadata"]["local_time_str"] = time;

	    // Do the same for the ut time
	    var time = record["metadata"]["ut_time"];
	    time = time.split(" ")[1].split(":",2).join(":");
	    record["metadata"]["ut_time_str"] = time;

	    // Get the image number from the filename
	    var imgnum = record["metadata"]["filename"];
	    var str_i = imgnum.indexOf(this.date_str);
	    if (str_i!=-1) {
	        var start = str_i+this.date_str.length;
	        imgnum = parseInt(imgnum.slice(start,start+4),10);
	    } else {
	        imgnum = "--";
	    }
	    record["metadata"]["image_number"] = imgnum;

	    // Turn the types list into a readable string
	    var obstype = record["metadata"]["instrument"];
	    var types = record["metadata"]["types"];
	    if (types.indexOf("ACQUISITION")!=-1) {
	        obstype += " acq";
	    }
	    obstype += " "+record["metadata"]["object"];
	    record["metadata"]["obstype"] = obstype;

	    // Format the wavelength into a more readable string
	    if (types.indexOf("SPECT")!=-1) {
		var wlen = record["metadata"]["wavelength"];
		wlen = parseInt(wlen) + "nm";
		record["metadata"]["wavelength_str"] = wlen;
	    } else {
		record["metadata"]["wavelength_str"] = 
		    record["metadata"]["filter"];
	    }

	    // Format some metrics into strings including errors
	    if (record["iq"]) {
		record["iq"]["delivered_str"] = 
	            record["iq"]["delivered"].toFixed(2) + " \u00B1 " +
		    record["iq"]["delivered_error"].toFixed(2);
		record["iq"]["zenith_str"] = 
		    record["iq"]["zenith"].toFixed(2) + " \u00B1 " +
		    record["iq"]["delivered_error"].toFixed(2);
	    }
	    if (record["cc"]) {
		record["cc"]["extinction_str"] = 
	            record["cc"]["extinction"].toFixed(2) + " \u00B1 " +
		    record["cc"]["extinction_error"].toFixed(2);

		// Average the reported zeropoints to get a single number
		var zp_dict = record["cc"]["zeropoint"];
		var zp=0, zperr=0, count=0;
		for (var key in zp_dict) {
		    zp += zp_dict[key]["value"];
		    zperr += Math.pow(zp_dict[key]["error"],2);
		    count++;
		}
		zp = zp / count;
		zperr = Math.sqrt(zperr);
		record["cc"]["zeropoint_str"] = zp.toFixed(2) +" \u00B1 " +
		                                zperr.toFixed(2);
	    }
	    if (record["bg"]) {
		record["bg"]["brightness_str"] = 
	            record["bg"]["brightness"].toFixed(2) + " \u00B1 " +
	            record["bg"]["brightness_error"].toFixed(2);
	    }

	    // Add the record to the database
	    var datalabel = record["metadata"]["datalabel"];
	    this.database.addRecord(datalabel, record);

	    // Replace the incoming record with the one from the
	    // database, in case it contained additional information
	    records[i] = this.database.getRecord(datalabel);
	} // end for-loop over records

	// Update table
	if (this.metrics_table) {

	    var key = [];

	    // Add in all column fields
	    for (var j in this.metrics_table.columns) {
		key.push(this.metrics_table.columns[j]["field"]);
		var swapval = this.metrics_table.columns[j]["swap"];
		if (swapval) {
		    key.push(swapval);
		}
	    }

	    // Add the record to the table
	    var table_records = this.formatTableRecords(records,key);
	    this.metrics_table.addRecord(table_records);

	    // Add warning icons to cells if necessary
	    var problem = '<span class="problem_icon"></span>';
	    var warn = '<span class="warn_icon"></span>';
	    var element, value;
	    for (var k in records) {
		var record = records[k];
		if (record["iq"]) {
		    if (record["iq"]["comment"].length>0) {
			element = $('#'+record["metadata"]["datalabel"]+
				  ' td.iq');
			value = element.text();

			if (record["iq"]["comment"].length==1 &&
			    record["iq"]["comment"][0].indexOf(
							"ellipticity")!=-1) 
			{
			    value = '<div class=outer>'+warn+value+'</div>';
			} else {
			    value = '<div class=outer>'+problem+value+'</div>';
			}
			element.html(value);
		    }
		}
		if (record["cc"]) {
		    if (record["cc"]["comment"].length>0) {
			element = $('#'+record["metadata"]["datalabel"]+
				  ' td.cc');
			value = element.text();
			value = '<div class=outer>'+problem+value+'</div>';
			element.html(value);
		    }
		}
		if (record["bg"]) {
		    if (record["bg"]["comment"].length>0) {
			element = $('#'+record["metadata"]["datalabel"]+
				  ' td.bg');
			value = element.text();
			value = '<div class=outer>'+problem+value+'</div>';
			element.html(value);
		    }
		}
	    }

	    // Add even class to even rows to allow them to be styled
	    // (unless they are highlighted)
	    $('#metrics_table tbody tr:odd').each(function(){
		$(this).removeClass("even");
	    }); // end each
	    $('#metrics_table tbody tr:even').each(function(){
	        if (!$(this).hasClass("highlight")) {
		    $(this).addClass("even");
		}
	    }); // end each

	    
	} else {
	    ////here -- what should happen?
	    console.log("No metrics table");
	}

	// Update plots
	var plot_record;
	if (this.iq_plot) {
	    var data_key = "iq-zenith";
	    var error_key = "iq-delivered_error";
	    plot_record = this.formatPlotRecords(records,data_key,error_key);
	    this.iq_plot.addRecord(plot_record);
	} else {
	    console.log("No IQ plot");
	}
	if (this.cc_plot) {
	    var data_key = "cc-extinction";
	    var error_key = "cc-extinction_error";
	    plot_record = this.formatPlotRecords(records,data_key,error_key);
	    this.cc_plot.addRecord(plot_record);
	} else {
	    console.log("No CC plot");
	}
	if (this.bg_plot) {
	    var data_key = "bg-brightness";
	    var error_key = "bg-brightness_error";
	    plot_record = this.formatPlotRecords(records,data_key,error_key);
	    this.bg_plot.addRecord(plot_record);
	} else {
	    console.log("No BG plot");
	}

	// Update tooltips
	var tt, tooltip_record;
	for (tt in this.tooltips) {
	    tooltip_record = this.formatTooltipRecords(records,tt);
	    this.tooltips[tt].addRecord(tooltip_record);
	}

	// Update the message in any currently displayed tooltips
	// with the message for the hovered-over row: this message
	// may have changed as the table updated under the mouse
	var mv = this;
	$("#metrics_table td").removeClass("hover");
	$("#metrics_table tr").each(function(){
	    if (mv.isHover($(this)) && mv.isHover($("#table_wrapper"))) {
		var dl = $(this).attr("id");
		for (tt in mv.tooltips) {
		    tt = mv.tooltips[tt];
		    var msg = tt.messages[dl];
		    $("#"+tt.id).text(msg);
		}

		$(this).find("td").each(function(){
		    if (mv.isHover($(this))) {
			$(this).addClass("hover");
		    }
		});

		return false;
	    }
	});

    }, // end update

    formatTableRecords: function(records, key) {

	var return_single = false;
	if (!(records instanceof Array)) {
	    records = [records];
	    return_single = true;
	}
	var tbl_records = [];
	for (var i in records) {
	    var record = records[i];
	    var table_record = {};
	    for (var j in key) {
	        var k = key[j].split("-",2);
	        var subdict = k[0];
	        var subkey = k[1];
		if (record[subdict]) {
		    table_record[key[j]] = record[subdict][subkey];
		} else {
		    table_record[key[j]] = "--";
		}
	    }
	    table_record["key"] = record["metadata"]["datalabel"];
	    
	    tbl_records.push(table_record);
	}

	if (return_single) {
	    return tbl_records[0];
	} else {
	    return tbl_records;
	}
    }, // end formatTableRecords
    
    formatPlotRecords: function(records, data_key, error_key) {
	var return_single = false;
	if (!(records instanceof Array)) {
	    records = [records];
	    return_single = true;
	}
	var plt_records = [];
	for (var i in records) {
	    var record = records[i];
	    var plot_record = {};

	    var dk = data_key.split("-",2);
	    var ek = error_key.split("-",2);

	    if (record[dk[0]]) {
		var time = record["metadata"]["local_time"];
		var value = record[dk[0]][dk[1]];
		var error = record[ek[0]][ek[1]];

		var series;
		if (dk[0]=="iq") {
		    series = record["metadata"]["waveband"];
		} else {
		    series = dk[0];
		}

		plot_record["series"] = series;
		plot_record["date"] = time;
		plot_record["data"] = value;
		plot_record["error"] = error;
	    
		plt_records.push(plot_record);
	    }
	}
	if (return_single) {
	    return plt_records[0];
	} else {
	    return plt_records;
	}
    }, // end formatPlotRecords
    
    formatTooltipRecords: function(records, key) {
	var return_single = false;
	if (!(records instanceof Array)) {
	    records = [records];
	    return_single = true;
	}
	var tt_records = [];
	for (var i in records) {
	    var record = records[i];
	    var tooltip_record = {};
	    var k = key.split("-",2);
	    tooltip_record["key"] = record["metadata"]["datalabel"];

	    if (record[k[0]]) {
		if (k[1]=="airmass") {
		    tooltip_record["message"] = "AM " + 
			                        record[k[0]][k[1]].toFixed(2);
		} else if (k[1]=="waveband") {
		    tooltip_record["message"] = record[k[0]][k[1]] + "-band";
		} else if (k[1]=="requested"){
		    tooltip_record["message"] = "Requested " +
			                        record[k[0]][k[1]];
		} else {
		    tooltip_record["message"] = record[k[0]][k[1]];
		}
	    } else {
		tooltip_record["message"] = "--";
	    }

	    tt_records.push(tooltip_record);
	}

	if (return_single) {
	    return tt_records[0];
	} else {
	    return tt_records;
	}
    }, // end formatTooltipRecords
    
    formatMessageRecords: function(records, key) {
	var return_single = false;
	if (!(records instanceof Array)) {
	    records = [records];
	    return_single = true;
	}

	var msg_records = [];
	for (var i in records) {
	    var record = records[i];
	    var message = "<span class=label> for image " + 
	                  record["metadata"]["image_number"] + ", " +
	                  record["metadata"]["datalabel"] + 
	                  ":</span> ";

	    var has_msg = false;
	    var subdicts = ["iq", "cc", "bg"];
	    for (var j in subdicts) {
		if (record[subdicts[j]]) {
		    var msg_array = record[subdicts[j]][key];
		    if (msg_array.length>0) {
			if (has_msg) {
			    message += ", ";
			}
			message += msg_array.join(", ");
			has_msg = true;
		    }
		}
	    }
	    if (!has_msg) {
		message += "(none)";
	    }

	    msg_records.push(message);
	}
	if (return_single) {
	    return msg_records[0];
	} else {
	    return msg_records;
	}
    }, // end formatMessageRecords

}; // end prototype
