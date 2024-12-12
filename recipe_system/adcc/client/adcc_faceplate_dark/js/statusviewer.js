// Driver class to instantiate all the viewports and hold the database
// of records

// Constructor
function StatusViewer(element, id) {
    this.element = element;
    this.id = id;

    // Placeholders for ViewPorts
    this.status_table = null;
    this.log_window = null;

    // Placeholder for database of records
    this.database = null;

    // Placeholder for adcc command pump
    this.gjs = null;

    // Load the site/time information from the adcc
    // and initialize the viewer
    this.load();
}
// Add methods to prototype
StatusViewer.prototype = {
    constructor: StatusViewer,
    load: function() {
	// Make an AJAX request to the server for the current
	// time and the server site information
	// The callback for this request will call the init function
	var sv = this;
	$.ajax({type: "GET",
		url: "/rqsite.json",
	        success: function (data) {
		    sv.site = data.local_site;
		    sv.tzname = data.tzname;
		    sv.utc_offset = data.utc_offset;

		    // Translate utc_now into a JS date
		    var udt = data.utc_now.split(" ");
		    var ud = udt[0].split("-");
		    var ut = udt[1].split(":");
		    ut[3] = (parseFloat(ut[2])-parseInt(ut[2],10))*1000;
		    ut[2] = parseInt(ut[2],10);

		    var ldate = new Date(Date.UTC(ud[0],ud[1]-1,ud[2],
						  ut[0],ut[1],ut[2], ut[3]));
		    // Add in the UT offset
		    sv.server_now =
			Date(ldate.setHours(ldate.getHours()+sv.utc_offset));

		    // Keep track of the difference between local
		    // time and server time
		    var ltz = ldate.getTimezoneOffset() / 60;
		    sv.tz_offset = sv.utc_offset - ltz;
		    sv.init();

                }, // end success
		error: function() {
		    sv.site = undefined;
		    sv.tzname = "LT";
		    sv.server_now = new Date();
		    sv.utc_offset = sv.server_now.getTimezoneOffset() / 60;
		    sv.tz_offset = 0;
		    sv.init();
		} // end error
	}); // end ajax

    }, // end load
    init: function() {

	// Reference to self, to use in functions inside init
	var sv = this;

	// Instantiate the records database
	this.database = new MetricsDatabase();

	// Check for a date parameter in the URL
	var datepar_regex = /^(.*)[?&]date=(\d{8})/;
	var url = $(location).attr('href');
	var datepar;
	if (url.match(datepar_regex)) {
	    datepar = url.replace(datepar_regex,'$2');
	} else {
	    datepar = undefined;
	}

	// If there is a date parameter, use it instead of the current time
	var prev_turnover;
	if (datepar) {
	    // This sets the previous turnover to UT 0, which is
	    // 14:00 HST or 20:00 (+/- an hour) Chile time.  Either
	    // way the following code to set it to 14:00 will
	    // get the right times for the fake UT used at
	    // either site.
	    var y = parseInt(datepar.slice(0,4),10);
	    var m = parseInt(datepar.slice(4,6),10)-1;
	    var d = parseInt(datepar.slice(6,8),10);
	    prev_turnover = new Date(Date.UTC(y,m,d));

	    // Set a variable indicating that we are in demo-mode
	    // and should not attempt to reset the viewer at 14:00
	    this.demo_mode = true;

	} else {
	    // Use the UT time passed by the adcc
	    prev_turnover = new Date(this.server_now);
	    this.demo_mode = false;
	}

	// Set the previous and next turnover times
	// Turnover is set to 14:00 at the server location
	var turntime = 14+this.tz_offset;
	if (prev_turnover.getHours()<turntime) {
	    prev_turnover.setDate(prev_turnover.getDate()-1);
	}
	prev_turnover.setHours(turntime);
	prev_turnover.setMinutes(0);
	prev_turnover.setSeconds(0);

	var next_turnover = new Date(prev_turnover.getFullYear(),
				     prev_turnover.getMonth(),
				     prev_turnover.getDate() + 1,
				     prev_turnover.getHours());

	// Store previous and next turnover to check against
	// in the update function
	this.prev_turnover = prev_turnover;
	this.turnover = next_turnover;

	// Add the basic structure to the element
	var html_str = this.composeHTML();
	this.element.html(html_str);

	// Add a date to the date box
	this.date_str = this.getDateString();
	$("#date").html(this.date_str);

	// Instantiate status table
	// Define table columns
	var columns = [
                       {id:"time", name:"Time taken",
			field:"metadata-local_time_str"},
                       {id:"filename", name:"Filename",
			field:"metadata-raw_filename"},
		       {id:"datalabel",	name:"Data Label",
			field:"metadata-datalabel"},
		       {id:"status", name:"Reduce Status",
			field:"status-current"}
		       ]; // end columns
	this.status_table = new SimpleTable($("#table_wrapper"),
					    "status_table",columns);

	// Instantiate log window
	this.log_window = new ViewPort($("#log_target"),"log_window");

	// Use previous turnover as initial timestamp (in UTC seconds)
	// for adcc query
	var timestamp = Math.round(prev_turnover.valueOf()/1000);

	// Hook up the adcc command pump to the update function
	this.gjs = new GJSCommandPipe();
	this.gjs.registerCallback("reduce_status",
				  function(msg){sv.update(msg);});
	this.gjs.startPump(timestamp,"reduce_status");

	// If not in demo mode, set up a timeout to check the time
	// every minute to see if the page needs to be turned over
	if (sv.demo_mode) {
	    sv.reset_timeout = null;
	} else {
	    sv.reset_timeout = setInterval(function(){
	        var current_time = new Date();
		if (current_time > sv.turnover) {
		    sv.reset();
		}
	    },60000);
	}

	// Hide the log window
	$("#log_wrapper").hide()

	// Make the status tab active
	$("#date_wrapper").addClass("active");
	$("#log_tab").addClass("inactive");

	// When log tab or status link is clicked, request the
	// specifed (or the latest available) log
	$(document).on("click","#log_tab,a.rqlog",function(){
	    $("#date_wrapper").removeClass("active").addClass("inactive");
	    $("#log_tab").removeClass("inactive").addClass("active");
	    $("#table_wrapper").hide();
	    $("#log_wrapper").show();

	    if ($(this).is("a")) {
		rqurl = $(this).attr("href");
	    } else {
		rqurl = "/rqlog.json?file=reducelog-latest";
	    }

	    $.ajax({type: "GET",
		    url: rqurl,
		    success: function (data) {sv.displayLog(data)},
		    error: function () {sv.displayLog()}
		});

	    return false;
        });

	// When date tab is clicked, show the status table
	$(document).on("click","#date_wrapper",function(){
	    $("#date_wrapper").removeClass("inactive").addClass("active");
	    $("#log_tab").removeClass("active").addClass("inactive");
	    $("#table_wrapper").show();
	    $("#log_wrapper").hide();
	    return false;
        });

    }, // end init

    composeHTML: function() {
	var html_str = "";

	// Outer wrapper
	html_str += '<div id='+this.id+' class="status_viewer">';

	// Date prefix box
	html_str += '<div id="date_wrapper" class="tab">'+
	            '<span class="label">Date prefix: </span>'+
	            '<span id="date"></span></div>';

	// Log tab
	html_str += '<div id="log_tab" class="tab"><span class="label">'+
	            'View Reduce Log</span></div>';

	// Display wrapper (holds either table or log window)
	html_str += '<div id="display_wrapper">';

	// Table wrapper
	html_str += '<div id="table_wrapper"></div>';

	// Log window wrapper
	html_str += '<div id="log_wrapper">' +
	            '<h2>Reduce Log</h2>' +
	            '<div id="log_target"></div>' +
	            '</div></div>';

	// End outer wrappers
	html_str += '</div></div>';

	return html_str;
    }, // end composeHTML

    getDateString: function() {
	var date_str = "";

	// Get prefix from server site information
	// If not GN or GS, don't use a prefix
	if (this.site=="gemini-north") {
	    date_str += "N";
	} else if (this.site=="gemini-south") {
	    date_str += "S";
	}

	var date = new Date(this.prev_turnover);

	// Timezone offset is in hours; 10 is Hawaii.
	var timezone = this.utc_offset;
	if (timezone<10) {
	    // Add one day, since the UT date we use at night is
	    // actually the UT date for the next day for any
	    // place east of Hawaii
	    date = new Date(date.getFullYear(),
			    date.getMonth(),date.getDate()+1);
	}

	var year = date.getUTCFullYear();

	// Pad month and day with 0 if necessary
	var month = date.getUTCMonth() + 1;
	month = (month <10 ? "0" : "") +  month;
	var day = date.getUTCDate();
	day = (day <10 ? "0" : "") +  day;

	date_str += year + month + day;
	if (this.site=="gemini-north" || this.site=="gemini-south") {
	    date_str += "S";
	}
	return date_str;
    },

    displayLog: function(data) {
	var log;
	if (!data || !data["log"]) {
	    log = "Log file not available";
	} else {
	    log = data["log"];
	}
	this.log_window.clearRecord();
	this.log_window.addRecord('<pre>'+log+'</pre>');
    },

    reset: function() {
	// Reset everything for date turnover

	// Stop the pump
	this.gjs.stopPump();

	// Clear the database
	this.database.clearDatabase();

	// Clear all ViewPorts
	this.status_table.clearRecord();
	this.log_window.clearRecord();

	// Set the turnover times
	var prev_turnover = this.turnover;
	var timestamp = Math.round(prev_turnover.valueOf()/1000);

	var next_turnover = new Date(prev_turnover.getFullYear(),
				     prev_turnover.getMonth(),
				     prev_turnover.getDate() + 1,
				     prev_turnover.getHours());
	this.prev_turnover = prev_turnover;
	this.turnover = next_turnover;

	// Restart the pump
	this.gjs.startPump(timestamp,"qametric");

    },

    update: function(records) {

	// Test input; make into an array if needed
	if (!records) {
	    return;
	}
	if (!(records instanceof Array)) {
	    records = [records];
	} else if (records.length==0) {
	    return;
	}

	var incoming = [];
	for (var i in records) {
	    var record = records[i];

	    // Add a few more useful fields to the record

	    // Replace the local time with a datetime string,
	    // derived from the ut_time field
	    // Input format: "YYYY-MM-DD HH:MM:SS.SSS"
	    var udt = record["metadata"]["ut_time"].split(" ");
	    var ud = udt[0].split("-");
	    var ut = udt[1].split(":");
	    ut[3] = (parseFloat(ut[2])-parseInt(ut[2],10))*1000;
	    ut[2] = parseInt(ut[2],10);

	    var ldate = new Date(Date.UTC(ud[0],ud[1]-1,ud[2],
					  ut[0],ut[1],ut[2], ut[3]));

	    // Skip this record if it is not within the current date
	    if (ldate<this.prev_turnover || ldate>this.turnover) {
		continue;
	    }

	    // Pad with zeroes if necessary
	    var month = ldate.getMonth() + 1;
	    month = (month <10 ? "0" : "") +  month;
	    var day = ldate.getDate();
	    day = (day <10 ? "0" : "") +  day;
	    var hour = ldate.getHours();
	    hour = (hour <10 ? "0" : "") +  hour;
	    var min = ldate.getMinutes();
	    min = (min <10 ? "0" : "") +  min;
	    var sec = ldate.getSeconds()+ldate.getMilliseconds()/1000.0;
	    sec = (sec <10 ? "0" : "") +  sec;

	    var ld = [ldate.getFullYear(),month,day];
	    var lt = [hour,min,sec];
	    record["metadata"]["local_time"] = ld.join("-") + " " +
	                                       lt.join(":");

	    // Get just the hours and minutes from the local/ut time
	    record["metadata"]["local_time_str"] = lt[0]+":"+lt[1];
	    record["metadata"]["ut_time_str"] = ut[0]+":"+ut[1];

	    // Strip any suffixes from the "raw" filename
	    var fn_regex = /.*((N|S)\d{8}S(\d{4}))(_[a-zA-Z0-9]+)?(\.fits?)?/;
	    var fn = record["metadata"]["raw_filename"];
	    if (fn.match(fn_regex)) {
		record["metadata"]["raw_filename"] =
		    fn.replace(fn_regex,'$1$5');
	    }

	    // Get the image number from the filename
	    var imgnum = record["metadata"]["raw_filename"];
	    if (imgnum.match(fn_regex)) {
		imgnum = parseInt(imgnum.replace(fn_regex,'$3'),10);
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
		wlen = parseFloat(wlen).toFixed(3) + "\u00B5m";
		record["metadata"]["wavelength_str"] = wlen;
	    } else {
		record["metadata"]["wavelength_str"] =
		    record["metadata"]["filter"];
	    }

	    // Add the record to the database
	    var datalabel = record["metadata"]["datalabel"];
	    this.database.addRecord(datalabel, record);

	    // Replace the incoming record with the one from the
	    // database, in case it contained additional information
	    incoming.push(this.database.getRecord(datalabel));

	} // end for-loop over records

	// Don't bother continuing if there are no valid records to process
	if (incoming.length<1) {
	    return;
	} else {
	    records = incoming;
	}

	// Update table
	if (this.status_table) {

	    var key = [];

	    // Add in all column fields
	    for (var j in this.status_table.columns) {
		key.push(this.status_table.columns[j]["field"]);
	    }

	    // Add the record to the table
	    var table_records = this.formatTableRecords(records,key);
	    this.status_table.addRecord(table_records);

	} else {
	    console.log("No status table");
	}

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
		    if (record[subdict][subkey]!=undefined) {

			// Add a link to the logfile if possible
			if (subkey=="current") {
			    if (record[subdict]["logfile"]) {
				var lf = encodeURIComponent(
				    record[subdict]["logfile"])
				table_record[key[j]] =
				    '<a class="rqlog" '+
				    'href="../rqlog.json?file='+lf+'">'+
				    record[subdict][subkey]+'</a>';
			    } else {
				table_record[key[j]] = record[subdict][subkey];
			    }
			} else {
			    table_record[key[j]] = record[subdict][subkey];
			}
		    } else {
			table_record[key[j]] = "--";
		    }
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

}; // end prototype
