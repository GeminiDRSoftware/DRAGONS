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
    this.lightbox = null;

    // Placeholder for database of records
    this.database = null;

    // Placeholder for adcc command pump
    this.gjs = null;

    // Load the site/time information from the adcc
    // and initialize the viewer
    this.load();
}
// Add methods to prototype
MetricsViewer.prototype = {
    constructor: MetricsViewer,
    load: function() {
    // Make an AJAX request to the server for the current
    // time and the server site information
    // The callback for this request will call the init function
    var mv = this;
    $.ajax({type: "GET",
        url: "/rqsite.json",
            success: function (data) {
            mv.site = data.local_site;
            mv.tzname = data.tzname;
            mv.utc_offset = data.utc_offset;

            // Translate utc_now into a JS date
            var udt = data.utc_now.split(" ");
            var ud = udt[0].split("-");
            var ut = udt[1].split(":");
            ut[3] = (parseFloat(ut[2])-parseInt(ut[2],10))*1000;
            ut[2] = parseInt(ut[2],10);

            var ldate = new Date(Date.UTC(ud[0],ud[1]-1,ud[2],
                          ut[0],ut[1],ut[2], ut[3]));
            // Add in the UT offset
            mv.server_now =
            Date(ldate.setHours(ldate.getHours()+mv.utc_offset));

            // Keep track of the difference between local
            // time and server time
            var ltz = ldate.getTimezoneOffset() / 60;
            mv.tz_offset = mv.utc_offset - ltz;
            mv.init();

                }, // end success
        error: function() {
            mv.site = undefined;
            mv.tzname = "LT";
            mv.server_now = new Date();
            mv.utc_offset = mv.server_now.getTimezoneOffset() / 60;
            mv.tz_offset = 0;
            mv.init();
        } // end error
    }); // end ajax

    }, // end load
    init: function() {

    // Reference to self, to use in functions inside init
    var mv = this;

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

    // Instantiate scrolling table
	// Define table columns

    var columns = [
        {id:"datetime", name:"Time", field:"metadata-local_time", width:56,
         sort:true, hidden:true, disable_search:true},
        {id:"obstype", name:"Obstype", field:"metadata-obstype", width:100,
         hidden:true},
        {id:"time", name:"LT", field:"metadata-local_time_str", width:50,
         swap: "metadata-ut_time_str", alt_name:"UT"},
        {id:"imgnum", name:"Img#", field:"metadata-image_number", width:30},
        {id:"datalabel", name:"Data Label", field:"metadata-datalabel", width:180},
        {id:"wlen", name:"  Wvlen", field:"metadata-wavelength_str", width:60},
        {id:"iq", name:"IQ", field:"iq-band_str", width:50},
        {id:"cc", name:"CC", field:"cc-band_str", width:100},
        {id:"bg", name:"BG", field:"bg-band_str", width:60},
        {id:"deliq", name:"Delivered IQ", field:"iq-delivered_str", width:90},
        {id:"zeniq", name:"Zenith IQ", field:"iq-zenith_str",width:90},
        {id:"strehl", name:"Strehl", field:"iq-strehl_str",width:60},
        {id:"zeropt", name:"Zeropoint", field:"cc-zeropoint_str",width:90},
        {id:"extinc", name:"Extinction", field:"cc-extinction_str",width:90},
        {id:"sky", name:"Sky Mag", field:"bg-brightness_str",width:90}
    ]; // end columns
    this.metrics_table = new ScrollTable($("#table_wrapper"),
                         "metrics_table",columns);

    // Instantiate message window
    this.message_window = new ViewPort($("#message_target"),"message_window");
    // Instantiate lightbox for urgent message display
    this.lightbox = new KeyedViewPort($("#lightbox_window"),"lightbox_message");
    $("#lightbox_window").prepend('<span class="close_icon"></span>'+
                      '<div><h2>WARNING</h2><hr></div>');

    // Instantiate plots

    // Set up necessary for all plots
    var mindate = new Date(this.prev_turnover);
    mindate.setHours(18);
    mindate.setMinutes(0);
    mindate.setSeconds(0);
    var maxdate = new Date(this.prev_turnover);
    maxdate.setHours(mindate.getHours()+13);
    maxdate.setMinutes(0);
    maxdate.setSeconds(0);

    var xlab = [this.date_str.slice(1,5),
            this.date_str.slice(5,7),
            this.date_str.slice(7,9)];
    xlab = xlab.join("-");

    options = {
        mindate: mindate.toString(),
        maxdate: maxdate.toString(),
        ymin: 0.0,
        ymax: null,
        invert_yaxis: false,
        overlay: [],
        series_labels: [""],
        series_colors: [""],
        bg_color: "black",
	color: "white",
        title: "",
        ut: false,
        ut_offset: this.utc_offset,
        timezone: this.tzname,
        xaxis_label: xlab,
        yaxis_label: ""};

    // IQ Plot
    var iq_options = $.extend(true,{},options);
    iq_options.title = "Zenith IQ";
    iq_options.ymax = 2.5;
    iq_options.yaxis_label = "Zenith IQ (arcsec)";
    iq_options.series_labels = ["u","g","r","i","Z","Y","X",
                     "J","H","K","L","M","N","Q"];

    // These colors were tested for distinctiveness under common
    // color-blindness conditions at http://newmanservices.com/colorblind
    // iq_options.series_colors = ["#3F35EA","#566AF5","#FF9E00","#9AB3FF",
    //              "#9CCF31","#C9E198","#F7E908","#F7F2A1",
    //              "#CE0000","#E64B4B","#86C7FF","#B9DFFF"];

        iq_options.series_colors = ["#3F35EA", "#5C84FF", "#C30000", "#FF9E00",
                   "#F7E908", "#9CCF99", "#D672E8", "#86B7FF",
                   "#C9E166", "#E64B4B", "#AA5E00", "#F7ff55",
                   "#9DDF00","#6C99FF"];

    // This got very long.  It should probably be moved somewhere else.
    iq_options.overlay = [
                  //u
                  [{y:0.60,name:"IQ20",color:'#888'},
                           {y:0.90,name:"IQ70",color:'#888'},
                           {y:1.20,name:"IQ85",color:'#888'}],
                  //g
                  [{y:0.60,name:"IQ20",color:'#888'},
                           {y:0.85,name:"IQ70",color:'#888'},
                           {y:1.10,name:"IQ85",color:'#888'}],
                  //r
                  [{y:0.50,name:"IQ20",color:'#888'},
                           {y:0.75,name:"IQ70",color:'#888'},
                           {y:1.05,name:"IQ85",color:'#888'}],
                  //i
                  [{y:0.50,name:"IQ20",color:'#888'},
                           {y:0.75,name:"IQ70",color:'#888'},
                           {y:1.05,name:"IQ85",color:'#888'}],
                  //Z
                  [{y:0.50,name:"IQ20",color:'#888'},
                           {y:0.70,name:"IQ70",color:'#888'},
                           {y:0.95,name:"IQ85",color:'#888'}],
                  //Y
                  [{y:0.40,name:"IQ20",color:'#888'},
                           {y:0.70,name:"IQ70",color:'#888'},
                           {y:0.95,name:"IQ85",color:'#888'}],
                  //X
                  [{y:0.40,name:"IQ20",color:'#888'},
                           {y:0.65,name:"IQ70",color:'#888'},
                           {y:0.90,name:"IQ85",color:'#888'}],
                  //J
                  [{y:0.40,name:"IQ20",color:'#888'},
                           {y:0.60,name:"IQ70",color:'#888'},
                           {y:0.85,name:"IQ85",color:'#888'}],
                  //H
                  [{y:0.40,name:"IQ20",color:'#888'},
                           {y:0.60,name:"IQ70",color:'#888'},
                           {y:0.85,name:"IQ85",color:'#888'}],
                  //K
                  [{y:0.35,name:"IQ20",color:'#888'},
                           {y:0.55,name:"IQ70",color:'#888'},
                           {y:0.80,name:"IQ85",color:'#888'}],
                  //L
                  [{y:0.35,name:"IQ20",color:'#888'},
                           {y:0.50,name:"IQ70",color:'#888'},
                           {y:0.75,name:"IQ85",color:'#888'}],
                  //M
                  [{y:0.35,name:"IQ20",color:'#888'},
                           {y:0.50,name:"IQ70",color:'#888'},
                           {y:0.70,name:"IQ85",color:'#888'}],
                  //N
                  [{y:0.34,name:"IQ20",color:'#888'},
                           {y:0.37,name:"IQ70",color:'#888'},
                           {y:0.45,name:"IQ85",color:'#888'}],
                  //Q
                  [{y:0.00,name:"IQ20",color:'#888'},
                           {y:0.00,name:"IQ70",color:'#888'},
                           {y:0.54,name:"IQ85",color:'#888'}]];

    this.iq_plot = new TimePlot($("#iq_plot_wrapper"),"iqplot",iq_options);

    // CC Plot
    var cc_options = $.extend(true,{},options);
    cc_options.title = "Cloud Extinction";
    cc_options.ymin = -0.2;
    cc_options.ymax = 3.0;
    cc_options.yaxis_label = "Extinction (mag)";
    cc_options.series_labels = ["cc"];
    cc_options.series_colors = ["#86C7FF"];
    cc_options.overlay = [[{y:0.08,name:"CC50",color:'#888'},
                           {y:0.3,name:"CC70",color:'#888'},
                           {y:1.0,name:"CC80",color:'#888'}]];
    this.cc_plot = new TimePlot($("#cc_plot_wrapper"),"ccplot",cc_options);

    // BG Plot
    var bg_options = $.extend(true,{},options);
    bg_options.ymin = 17.0;
    bg_options.ymax = 24.0;
    bg_options.type = "optical";
    bg_options.title = "Sky Brightness";
    bg_options.title = "Sky Brightness <button type=\"button\" id=\"opt_ir_switch\">Show IR</button>";
    bg_options.invert_yaxis = true;
    bg_options.yaxis_label = "Sky Brightness (mag/arcsec^2)";
    bg_options.series_labels = ["u","g","r","i","Z"];
    bg_options.series_colors = ["#3f35ea","#5C84FF","#C30000",
                                     "#FF9E00","#F7E908"];
        bg_options.overlay = [ //u
                           [{y:21.66,name:"BG20",color:'#888'},
                            {y:19.49,name:"BG50",color:'#888'},
                            {y:17.48,name:"BG80",color:'#888'}],
                   //g
                           [{y:21.62,name:"BG20",color:'#888'},
                            {y:20.68,name:"BG50",color:'#888'},
                            {y:19.36,name:"BG80",color:'#888'}],
                   //r
                           [{y:21.33,name:"BG20",color:'#888'},
                            {y:20.32,name:"BG50",color:'#888'},
                            {y:19.34,name:"BG80",color:'#888'}],
                   //i
                           [{y:20.44,name:"BG20",color:'#888'},
                            {y:19.97,name:"BG50",color:'#888'},
                            {y:19.30,name:"BG80",color:'#888'}],
                   //Z
                           [{y:19.51,name:"BG20",color:'#888'},
                            {y:19.04,name:"BG50",color:'#888'},
                            {y:18.37,name:"BG80",color:'#888'}],
                  ];
    this.bg_plot = new TimePlot($("#bg_plot_wrapper"),"bgplot",bg_options);

    // Instantiate tooltips
    this.tooltips = {};

    // Ellipticity, for delivered IQ table column
    var el_tt = new TooltipOverlay($("#tooltip_wrapper"),
                       "tooltip_ellipticity",
                       "#metrics_table td.deliq");
    this.tooltips["iq-ellipticity_str"] = el_tt;

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
        this.tooltips[met+"-requested_str"] = tt;
    }

    // Full filename, for image number column
    var fn_tt = new TooltipOverlay($("#tooltip_wrapper"),"tooltip_filename",
                       "#metrics_table td.imgnum");
    this.tooltips["metadata-raw_filename"] = fn_tt;

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
        $("#metrics_table td").removeClass("hover");
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

    // Add event handler to link message window and plot highlighting
    // to clicks in table rows
    $("#metrics_table tbody").on("click", "tr", function(ev,from_plot) {
        var selected = $(this).hasClass("highlight");

        // Get some position information so we can scroll to the
        // row clicked if not visible
        var dl = $(this).find("td.datalabel").text();
        var tbody = $("#metrics_table tbody");
        var row = $(this);
        var view_top = tbody.offset().top;
        var view_btm = view_top + tbody.height();
        var row_top = row.offset().top;
        var row_btm = row_top + row.height();
        var in_scroll = row_btm<view_btm && row_top>view_top;
        var visible = row.hasClass("visible");

        if (selected && from_plot) {
        // If already selected, but click is from plot,
        // scroll row into view, leave everything else as is
        if (visible && !in_scroll) {
            tbody.stop().animate({
                scrollTop: row_top-view_top+tbody.scrollTop()
            },500);
        }
        } else {

        // clear previous selections
        $("#metrics_table tbody tr").removeClass("highlight");
        mv.message_window.clearRecord();
        $("#problem_overlay").hide();
        $("#warning_overlay").hide();
        if (!from_plot) {
            mv.iq_plot.highlightPoint();
            mv.cc_plot.highlightPoint();
            mv.bg_plot.highlightPoint();
        }

        if (!selected) {
            // add new selection (if not deselecting)

            $(this).addClass("highlight");
            var record = mv.database.getRecord(dl);
            var msg = mv.formatMessageRecords(record,"comment");
                mv.message_window.addRecord(msg);

            // Define a couple effects to perform
            // if the event was not triggered by one of the plots
            // (ie. a direct click or a lightbox-clear action)
            function highlight_effects() {
            if (!from_plot) {
                // Flash a red or yellow border around the message
                // box if there is a problem or warning
                if (row.find(".problem_icon").length>0) {
                $("#warning_overlay").hide();
                $("#problem_overlay").stop()
                    .fadeIn().fadeOut("slow");
                } else if (row.find(".warn_icon").length>0) {
                $("#problem_overlay").hide();
                $("#warning_overlay").stop()
                    .fadeIn().fadeOut("slow");
                }

                // Highlight the associated point in the plot
                mv.iq_plot.highlightPoint(dl);
                mv.cc_plot.highlightPoint(dl);
                mv.bg_plot.highlightPoint(dl);
            }
            }

            // If the row is not within view, scroll to it,
            // then call the highlight effects
            if (visible && !in_scroll) {
                tbody.stop().animate({
                scrollTop: row_top-view_top+tbody.scrollTop()
                }, 500, highlight_effects);
            } else {
            // Otherwise just call the highlight effects
            highlight_effects();
            }
        }
        }
        // Wait a bit, then clear the clear_warning flag
        var timeout = setTimeout(function() {
        mv.clear_warning = undefined;
        }, 800);
        return false;
    }); // end click

    // Add a handler to do the same when a point is highlighted
    // in the plot (ie. moused-over)
    $("#plot_wrapper").on("jqplotDataPointHighlight","div.jqplot-target",
        function(ev,pt) {
        // If mouse is over plots (ie. highlight came from mouse-over),
        // then clear any other highlighted points and highlight
        // the associated table row
        if (mv.isHover($("#plot_wrapper")) && !mv.clear_warning) {
            var plotname = $(this).attr("id");
            if (plotname=="iqplot") {
            mv.cc_plot.highlightPoint();
            mv.bg_plot.highlightPoint();
            } else if (plotname=="ccplot") {
            mv.iq_plot.highlightPoint();
            mv.bg_plot.highlightPoint();
            } else {
            mv.iq_plot.highlightPoint();
            mv.cc_plot.highlightPoint();
            }
            var dl = pt.data[2];
            var row = $("#"+dl);
            var from_plot = true;
            row.trigger("click",from_plot);
        }
        return false;
        }
        );
    $("#plot_wrapper").on("jqplotDataPointUnhighlight","div.jqplot-target",
        function() {
            $("#metrics_table tr").removeClass("highlight");
        return false;
        }
    );

    // Add a handler to link LT/UT column swap to LT/UT plot swap
    $("#metrics_table").on("swapColumn", "th.time", function() {
        var plots = [mv.iq_plot,mv.cc_plot,mv.bg_plot];
        for (var this_plot in plots) {
        this_plot = plots[this_plot];
        if ($(this).text()=="UT") {
            this_plot.options["ut"] = true;
        } else {
            this_plot.options["ut"] = false;
        }
        this_plot.addRecord();
        }
    });

    // Add a handler to hide the lightbox effect when there is a click
    // anywhere else in the window
    $(document).on("click",
               "#lightbox_background,#lightbox_window span.close_icon",
               function(){

        // Set a flag to make row/plot highlighting behave properly
        mv.clear_warning = true;

        // Get the datalabel for the top warning message
        var dl = $("#lightbox_message span.datalabel:first").text();

        mv.lightbox.clearRecord();
        $("#lightbox_background,#lightbox_window").hide();

        // Wait a bit, then highlight the row associated with the
        // top warning message
        var timeout = setTimeout(function() {
            $("#"+dl).click();
        }, 100);
    });

    // Add handler to change appearance of buttons in control panel
    // when selected
    $("#controls_wrapper").on("mousedown","li",function(){
        $(this).addClass("selected");
    });
    $("#controls_wrapper").on("mouseup","li",function(){
        $(this).removeClass("selected");
    });

    // Add handler to restore viewer to defaults when reset
    // button is clicked
    $("#controls_wrapper").on("click","#reset",function(){
        mv.restore();
    });

    // Add handler to pop up window showing reduction status
    $("#controls_wrapper").on("click","#status",function(){
        var props = "height=400,width=600,resizable=yes,scrollbars=yes";
        var new_win;
        if (datepar) {
        new_win = open("/qap/reduce_status.html?date="+datepar,
                   "Reduction Status",props);
        } else {
        new_win = open("/qap/reduce_status.html",
                   "Reduction Status",props);
        }
        return false;
    });

    // Add handler to show help message when help button is clicked
    // and to hide it when either help button is clicked again or
    // window is closed
    $("#controls_wrapper").on("click","#help",function(){
        mv.help();
        });
    $("#"+this.id).on("click", "#help_window span.close_icon", function(){
        mv.help();
        });

    // Add handler to switch bg plots from optical to IR
        $(document).on("click", "#opt_ir_switch", function() {
        if (bg_options.type == "optical") {
                bg_options.ymin = 9.0;
                bg_options.ymax = 19.0;
                bg_options.type = "ir";
                bg_options.title = "Sky Brightness <button type=\"button\" id=\"opt_ir_switch\">Show optical</button>";
                bg_options.series_labels = ["Y","X","J","H","K","L","M"];
          bg_options.series_colors = ["#9CCF99","#D672E8", "#86B7FF","#C9E166",
                           "#E64B4B", "#AA5E00", "#F7ff55"]

             //bg_options.series_colors = ["#3F35EA","#5C84FF","#C30000",
            //                   "#FF9E00","#F7E908","#3f35ea"];
                bg_options.overlay = [
            // J
                  [{y:16.2,name:"BGAny",color:'#888'}],
              // H
                  [{y:13.8,name:"BGAny",color:'#888'}],
              // K
                  [{y:14.6,name:"BGAny",color:'#888'}],
          ];
            } else if (bg_options.type == "ir") {
                bg_options.ymin = 17.0;
                bg_options.ymax = 24.0;
                bg_options.type = "optical";
                bg_options.title = "Sky Brightness <button type=\"button\" id=\"opt_ir_switch\">Show IR</button>";
              bg_options.series_labels = ["u","g","r","i","Z"];
            bg_options.series_colors = ["#3f35ea","#5C84FF","#C30000",
                                            "#FF9E00","#F7E908"];
                bg_options.overlay = [
             // u
                [{y:21.66,name:"BG20",color:'#888'},
                 {y:19.49,name:"BG50",color:'#888'},
                 {y:17.48,name:"BG80",color:'#888'}],
            // g
                [{y:21.62,name:"BG20",color:'#888'},
                 {y:20.68,name:"BG50",color:'#888'},
                 {y:19.36,name:"BG80",color:'#888'}],
            // r
                [{y:21.33,name:"BG20",color:'#888'},
                 {y:20.32,name:"BG50",color:'#888'},
                 {y:19.34,name:"BG80",color:'#888'}],
              // i
                  [{y:20.44,name:"BG20",color:'#888'},
                 {y:19.97,name:"BG50",color:'#888'},
                   {y:19.30,name:"BG80",color:'#888'}],
              // Z
                [{y:19.51,name:"BG20",color:'#888'},
                   {y:19.04,name:"BG50",color:'#888'},
                   {y:18.37,name:"BG80",color:'#888'}],
          ];
            }
        mv.bg_plot = new TimePlot($("#bg_plot_wrapper"),"bgplot",bg_options);
        mv.restore();
        });


    // Use previous turnover as initial timestamp (in UTC seconds)
    // for adcc query
    var timestamp = Math.round(prev_turnover.valueOf()/1000);

    // Set up a mouse-position tracker
    this.last_pos = {x:0,y:0};
    $(document).mousemove(function(e) {
        mv.last_pos = {x:e.pageX,y:e.pageY};
    });

    // Hook up the adcc command pump to the update function
    this.gjs = new GJSCommandPipe();
    this.gjs.registerCallback("qametric",function(msg){mv.update(msg);});
    this.gjs.startPump(timestamp,"qametric");

    // If not in demo mode, set up a timeout to check the time
    // every minute to see if the page needs to be turned over
    if (mv.demo_mode) {
        mv.reset_timeout = null;
    } else {
        mv.reset_timeout = setInterval(function(){
            var current_time = new Date();
        if (current_time > mv.turnover) {
            mv.reset();
        }
        },60000);
    }
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
    html_str += '<div id="message_wrapper">' +
                '<div id="problem_overlay"></div>' +
                '<div id="warning_overlay"></div>' +
                '<div id="message_box">'+
                '<div class="label">Messages&nbsp;</div>' +
                '<div id="message_target"></div>' +
                '</div></div>';

    // Plot wrapper
    html_str += '<div id="plot_wrapper">'+
                '<div id="iq_plot_wrapper"></div>'+
                '<div id="cc_plot_wrapper"></div>'+
                '<div id="bg_plot_wrapper"></div>'+
                '</div>';

    // Control bar
    html_str += '<div id="controls_wrapper">'+
                '<ul id="controls">'+
                '<li id="status" class="control">Reduce Status</li>'+
                '<li id="reset" class="control">Reset</li>'+
                '<li id="help" class="control">Help</li>'+
                '</ul>'+
                    '</div>';

    // Tooltip wrapper
    html_str += '<div id="tooltip_wrapper"></div>';

    // Lightbox wrapper
    html_str += '<div id="lightbox_wrapper">'+
                '<div id=lightbox_background></div>'+
                '<div id=lightbox_window></div>'+
                '</div>';

    // End outer wrapper
    html_str += '</div>';

    return html_str;
    }, // end composeHTML

    getBandString: function(metric,band) {
    var band_str = "";

    if (!band) {
        band_str = null;
    } else {
        if (band instanceof Array && band.indexOf(100)!=-1) {
        band[band.indexOf(100)] = "Any";
        } else if (band==100) {
        band = "Any";
        }
        band_str = metric.toUpperCase() + band;
    }

    return band_str;
    }, // end getBandString

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

    help: function() {
    var help_window = $("#help_window");
    if (help_window.length>0) {
        help_window.remove();
    } else {
        var help_msg =
        '<div id="accordion_wrapper">\
             <h2>Nighttime QA Metrics Help</h2><hr>\
             <div id=accordion>\
             <h3><span class="icon expand_icon"></span>What to Expect</h3>\
             <div class="text">\
             <p>As the Quality Assurance Pipeline (QAP) reduces incoming data\
             it generates quality assurance metrics: image quality, cloud\
             cover, and sky background measurements.  It reports these \
             numbers to a server, which automatically feeds them to\
             this interface.</p>\
             <p>When a frame has been reduced by the QAP, you should see a \
             new row appear in the table, displaying its measured QA metrics.\
             A new point will also appear in at least one of the plots.</p>\
             <p>If there is a problem with an observation, a \
             <span class="problem">WARNING</span> message\
             will pop up; click to clear it. The observation that generated\
             the warning will be highlighted, and the warning message will\
             appear in the message window in the middle of the page.</p>\
             <p>In the IQ plot, the zenith-corrected FWHM measurements are\
             sorted into photometric bands (like U, B, V, R, I, etc.), since\
             each band has a different definition of the acceptable percentile\
             bands.  Likewise, the sky brightness is sorted by filter.\
             No sorting is necessary for cloud extinction.</p>\
             </div>\
             <h3><span class="icon expand_icon"></span>Interactions to Try</h3>\
             <div class="text">\
             <p>In the table:</p>\
             <ul>\
             <li>Click on a row to see associated warnings and messages.\
                 This will also highlight the associated data point(s) in\
                 the plots.</li>\
             <li>Mouse-over the entries to see more information about the\
                 observation and/or the QAP measurements.</li>\
             <li>Filter the table entries by entering any string into the\
                 search box below the table.  Any text visible in the table\
                 is searchable, as is the observation information in the\
                 pop-up over the Data Label field.  For example, you can\
                 enter &quot;gmos&quot; to show only gmos observations,\
                 or &quot;q-13&quot; to show only observations with Q-13\
                 in the data label.</li>\
             <li>Rearrange the columns by dragging and dropping the \
                 column headers.</li>\
             <li>Switch from LT to UT by clicking on the LT column header.\
                 This will also switch the time format on the plots.</li>\
             </ul>\
             <p>In the plots:</p>\
             <ul>\
             <li>Mouse-over data points to get the associated data label\
                 and pull up the conditions limits (eg. lines indicating\
                 where IQ20, IQ70, and IQ85 end).  This will also \
                 highlight the associated row in the table, and\
                 show its warnings in the message window.</li>\
             <li>Zoom in to a box by clicking and dragging; double-click\
                 to zoom out.</li>\
             <li>Select a series to hide or unhide by clicking on the \
                 legend.</li>\
             </ul>\
             <p>In the control panel:</p>\
             <ul>\
             <li>Click Reduce Status to open a window showing the \
                 status of recently reduced files, and to view \
                 more detailed logs from these reductions.</li> \
             <li>Click Reset to restore the table and the plots to\
                 the default configuration.</li>\
             <li>Click Help to display or hide this message.</li>\
             </ul>\
             </div>\
             <h3><span class="icon expand_icon"></span>Troubleshooting</h3>\
             <div class="text">\
             <p>If you are experiencing difficulties with this interface,\
                try these steps to recover normal behavior:</p>\
             <ul>\
             <li>If new observations are not appearing, check the health\
                 of the QAP or its server.</li>\
             <li>If you are seeing strange behavior, try reloading the page.\
                 Nothing will be lost: all data will be immediately \
                 recovered from the server.</li>\
             <li>File a fault report against the QAP.</li>\
             </ul>\
             </div>\
             </div></div>\
            ';

        $("#"+this.id).append('<div id="help_window"></div>');
        $("#help_window").html(help_msg)
                         .prepend('<span class="close_icon"></span>')
                         .draggable()
                         .resizable({handles: 'se',
                                         minWidth: 180,
                             minHeight: 180,
                     start: function(){
                         $(this).data("resized",true)
                         }});
        $("#accordion div.text").hide();
        $("#accordion span.icon").click(function() {
        $(this).parent().next().toggle();
        $(this).toggleClass("expand_icon")
               .toggleClass("unexpand_icon");
        if (!$("#help_window").data("resized")) {
            if ($("#accordion span.unexpand_icon").length==0) {
            $("#help_window").css("height","200px");
            } else {
            $("#help_window").css("height","400px");
            }
        }
        return false;
            });
    }
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

    // Clear all ViewPorts
    this.metrics_table.clearRecord();
    this.message_window.clearRecord();
    this.iq_plot.clearRecord();
    this.cc_plot.clearRecord();
    this.bg_plot.clearRecord();
    for (var tt in this.tooltips) {
        this.tooltips[tt].clearRecord();
    }
    this.lightbox.clearRecord();
    $("#lightbox_background,#lightbox_window").hide();
    $("#help_window").remove();

    // Set the turnover times
    var prev_turnover = this.turnover;
    var timestamp = Math.round(prev_turnover.valueOf()/1000);

    var next_turnover = new Date(prev_turnover.getFullYear(),
                     prev_turnover.getMonth(),
                     prev_turnover.getDate() + 1,
                     prev_turnover.getHours());
    this.prev_turnover = prev_turnover;
    this.turnover = next_turnover;

    // Update date string
    this.date_str = this.getDateString();
    $("#date").html(this.date_str);

    // Update min/max date on plots
    var mindate = new Date(this.prev_turnover);
    mindate.setHours(18);
    mindate.setMinutes(0);
    mindate.setSeconds(0);
    var maxdate = new Date(this.prev_turnover);
    maxdate.setHours(mindate.getHours()+13);
    maxdate.setMinutes(0);
    maxdate.setSeconds(0);
    var xlab = [this.date_str.slice(1,5),
            this.date_str.slice(5,7),
            this.date_str.slice(7,9)];
    xlab = xlab.join("-");

    this.iq_plot.updateDate(mindate,maxdate,xlab);
    this.cc_plot.updateDate(mindate,maxdate,xlab);
    this.bg_plot.updateDate(mindate,maxdate,xlab);

    // Restart the pump
    this.gjs.startPump(timestamp,"qametric");

    },

    restore: function() {
    // Restore all ViewPorts to default state, with current data

    var records = this.database.getRecordList();

    // Clear all ViewPorts
    this.metrics_table.clearRecord();
    this.message_window.clearRecord();
    this.iq_plot.clearRecord();
    this.cc_plot.clearRecord();
    this.bg_plot.clearRecord();
    for (var tt in this.tooltips) {
        this.tooltips[tt].clearRecord();
    }
    this.lightbox.clearRecord();
    $("#lightbox_background,#lightbox_window").hide();
    $("#help_window").remove();

    var disable_warning = true;
    this.update(records,disable_warning);

    },

    update: function(records, disable_warning) {

    var mv = this;

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
    var incoming_metric = {};
    for (var i in records) {
        var record = records[i];

        // Wrap this whole section in a try/except clause. It
        // isn't a pretty way of doing things, but it can catch
        // engineering data with missing metrics without stopping
        // the whole GUI for a night and inconveniencing the
        // observers.
        try {

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
        ldate.setHours(ldate.getHours()-this.tz_offset);

        // Skip this record if it is not within the current date
        var remote_start = new Date(this.prev_turnover);
        remote_start.setHours(remote_start.getHours()-this.tz_offset);
        var remote_end = new Date(this.turnover);
        remote_end.setHours(remote_end.getHours()-this.tz_offset);
        if (ldate<remote_start || ldate>remote_end) {
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
        if (types.indexOf("SPECT")!=-1) {
                obstype += " spect";
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

        // Format some metrics into strings including errors
        // and format integer bands into strings
        // (eg. 50 -> "CC50")
        var datalabel = record["metadata"]["datalabel"];
        if (incoming_metric[datalabel]==undefined) {
            incoming_metric[datalabel] = [];
        }

        if (record["iq"]) {
            incoming_metric[datalabel].push("iq");

            // If the observation uses AO, there may not be a
            // delivered IQ
            try {
            record["iq"]["delivered_str"] =
                record["iq"]["delivered"].toFixed(2) + " \u00B1 " +
                record["iq"]["delivered_error"].toFixed(2);
            }
            catch (e) {
            record["iq"]["delivered_str"] = undefined;
            }

            // FITSStore may not deliver zenith metrics
            if (record["iq"]["zenith"] && record["iq"]["zenith_error"]) {
            record["iq"]["zenith_str"] =
                    record["iq"]["zenith"].toFixed(2) + " \u00B1 " +
                record["iq"]["zenith_error"].toFixed(2);
            } else if (record["iq"]["zenith"] && !(record["iq"]["zenith_error"])) {
                // AO seeing corrected is iq-zenith
                if (record["iq"]["is_ao"]) {
                    record["iq"]["zenith_str"] =
                        record["iq"]["zenith"].toFixed(2) + " (AO)";
                } else {
                    record["iq"]["zenith_str"] =
                        record["iq"]["zenith"].toFixed(2);
                    }
            } else {
            record["iq"]["zenith_str"] = undefined;
            }

            // If an AO seeing value is provided, use this
            if (record["iq"]["ao_seeing_zenith"]) {
            record["iq"]["zenith_str"] =
                record["iq"]["ao_seeing_zenith"].toFixed(2) + " (AO)";

            // Add a hack here to overwrite the zenith IQ if
            // there is an AO-estimated seeing value, this is
            // the value that will then be plotted
                record["iq"]["zenith"] =
                    record["iq"]["ao_seeing_zenith"];

            // Putting in zero errors is obviously untrue, but prevents
            // formatPlotRecords from throwing out the data
            record["iq"]["zenith_error"] = 0.0;
            }

            try {
            record["iq"]["strehl_str"] =
                record["iq"]["strehl"].toFixed(2);
            }
            catch (e) {
                record["iq"]["strehl_str"] = undefined;
            }

            if (record["iq"]["ellipticity"]) {
            record["iq"]["ellipticity_str"] =
                record["iq"]["ellipticity"].toFixed(2) + " \u00B1 " +
                record["iq"]["ellip_error"].toFixed(2);
            }
            record["iq"]["band_str"] =
            this.getBandString("iq",record["iq"]["band"]);
            record["iq"]["requested_str"] =
            this.getBandString("iq",record["iq"]["requested"]);
        }
        if (record["cc"]) {
            incoming_metric[datalabel].push("cc");
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
            record["cc"]["zeropoint_str"] = zp.toFixed(2) +" \u00B1 " + zperr.toFixed(2);
            record["cc"]["band_str"] =
            this.getBandString("cc",record["cc"]["band"]);
            record["cc"]["requested_str"] =
            this.getBandString("cc",record["cc"]["requested"]);
        }
        if (record["bg"]) {
            incoming_metric[datalabel].push("bg");
            record["bg"]["brightness_str"] =
            record["bg"]["brightness"].toFixed(2) + " \u00B1 " +
            record["bg"]["brightness_error"].toFixed(2);
            record["bg"]["band_str"] =
            this.getBandString("bg",record["bg"]["band"]);
            record["bg"]["requested_str"] =
            this.getBandString("bg",record["bg"]["requested"]);
        }

        // Catch for missing metrics
        } catch (e) {
        var datalabel = record["metadata"]["datalabel"];
        if (incoming_metric[datalabel]==undefined) {
            incoming_metric[datalabel] = "No datalabel";
        }

        }


        // Add the record to the database
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
    if (this.metrics_table) {
        // Get the scroll position before updating, so it can be
        // restored later (Firefox 3.6 likes to reset it to 0)
        var scrolltop = $("#metrics_table tbody").scrollTop();

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

        // Add a pop-up box to show what is in data stacks
        for (var k=0; k<records.length; k++) {
        var record = records[k];
        var datalabel = record["metadata"]["datalabel"];
        var stack = record["metadata"]["stack"];
        if (stack) {
            // Locate the table cells that contain BG and sky magnitude as
            // these are not measured for stacked data
            var jquery_results = $('tr#'+datalabel).find('td.bg')
            var cell = jquery_results[0];
            $(cell).html('N/A');
            var jquery_results = $('tr#'+datalabel).find('td.sky')
            var cell = jquery_results[0];
            $(cell).html('N/A');
            // Locate the table cells that contain stacked data
            var jquery_results = $('tr#'+datalabel).find('td.datalabel')
            var cell = jquery_results[0];
            var cell_value = cell.innerHTML;
            // Style those table cells as links to make them obvious
            var div_regex = /^<div class="stack_link">/;
            if (!(cell_value.match(div_regex))) {
            $(cell).html('<div class="stack_link">'+cell_value+'</div>')
            }
            $(cell).click({datalabel:datalabel, stack:stack}, function(event) {
            // Format the stack list for display
            var stack_str = "";
            for (var m=0; m<event.data.stack.length; m++) {
                stack_str += event.data.stack[m] + '<br />'
            }
            var stack_msg = '<div class="stack_wrapper">\
                        <h3>Stack list for ' + event.data.datalabel + '</h3><hr>\
                        <div id=stack_list>\
                        <div class="text">' + stack_str + '</div></div></div>'
            // Add a stack list window
            var stack_id = "stack_window_" + event.data.datalabel
            if ($('#'+stack_id).length != 0) {
                $('#'+stack_id).remove();
            }
            $("#"+mv.id).append('<div id='+stack_id+' class="stack_window" resizable=yes></div>');
            var close_id = "close_window_" + event.data.datalabel
            var close_button = $('<span>', {'class': 'close_icon', 'id': close_id})
            close_button.click(function() {
                $('#'+stack_id).remove();
            })
            $("#"+stack_id).html(stack_msg)
            .prepend(close_button)
                    .draggable()
                    .resizable({handles: 'se',
                minWidth: 180,
                minHeight: 180,
//                start: function(){
//                    $(this).data("resized",true)
//                    $('#'+stack_id).data("resized",true)
//                }
            });
            // End of cell click event
            })
        // End of stack loop
        }
        // End of loop over records
        };

        // Add warning icons to cells if necessary
        var problem = '<span class="problem_icon"></span>';
        var warn = '<span class="warn_icon"></span>';
        var element, value;
        var problem_records = {};
        problem_records["size"] = 0;
        problem_records["index"] = {};
        for (var k=0;k<records.length;k++) {
        var record = records[k];
        var found_problem = false;
        var datalabel =  record["metadata"]["datalabel"];
        if (record["iq"]) {
            if (record["iq"]["comment"].length>0) {
            element = $('#'+datalabel+' td.iq');
            value = element.text();

            if ( record["iq"]["comment"].length==1 &&
                 (record["iq"]["comment"][0].indexOf("ellipticity")!=-1 ||
                  record["iq"]["comment"][0].indexOf("Single source")!=-1 ||
		  record["iq"]["comment"][0].indexOf("estimated AO seeing")!=-1 ||
                  record["metadata"]["types"].indexOf("SPECT")!=-1 ) )
            {
                value = '<div class=outer>'+warn+value+'</div>';
            } else {
                value = '<div class=outer>'+problem+value+'</div>';

                if (incoming_metric[datalabel].indexOf("iq")!=-1) {
                found_problem = true;
                }
            }
            element.html(value);
            }
        }
        if (record["cc"]) {
            if (record["cc"]["comment"].length>0) {
            element = $('#'+datalabel+' td.cc');
            value = element.text();
            value = '<div class=outer>'+problem+value+'</div>';
            element.html(value);
            if (incoming_metric[datalabel].indexOf("cc")!=-1) {
                found_problem = true;
            }
            }
        }
        if (record["bg"]) {
            if (record["bg"]["comment"].length>0) {
            element = $('#'+datalabel+' td.bg');
            value = element.text();
            value = '<div class=outer>'+problem+value+'</div>';
            element.html(value);
            if (incoming_metric[datalabel].indexOf("bg")!=-1) {
                found_problem = true;
            }
            }
        }

        if (found_problem) {
            problem_records[datalabel] = record;
            problem_records.index[datalabel] = k;
            problem_records.size++;
        }
        }

        // Restore scroll position
        $("#metrics_table tbody").scrollTop(scrolltop);

    } else {
        console.log("No metrics table");
    }

    // Update plots
    var plot_record;
    if (this.iq_plot) {
        var data_key = "iq-zenith";
        var error_key = "iq-zenith_error";
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

    // If there is a table row highlighted, re-highlight its
    // associated point in the plots
    var dl = $("#metrics_table tr.highlight").attr("id");
    if (dl) {
        if (!this.isHover($("#plot_wrapper"))) {
        this.iq_plot.highlightPoint(dl);
        this.cc_plot.highlightPoint(dl);
        this.bg_plot.highlightPoint(dl);
        $("#"+dl).addClass("highlight");
        } else {
        this.iq_plot.highlightPoint();
        this.cc_plot.highlightPoint();
        this.bg_plot.highlightPoint();
        }
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
    $("#metrics_table tbody td").removeClass("hover");
    $("#metrics_table tbody tr").each(function(){
        if (mv.isHover($(this)) && mv.isHover($("#table_wrapper"))) {
        var dl = $(this).attr("id");
        for (tt in mv.tooltips) {
            tt = mv.tooltips[tt];
            var msg = tt.messages[dl];
            $("#"+tt.id).text(msg);
        }

        $(this).find("td").not(".hidden").each(function(){
            if (mv.isHover($(this))) {
            $(this).addClass("hover");
            }
        });

        return false;
        }
    });

    // If problems were found in the incoming record(s), pull up
    // a lightbox displaying their messages
    if (!disable_warning && problem_records.size>0) {
        var pr = [];
        for (var r in problem_records) {
        if (r=="size" || r=="index") {
            continue;
        }
        pr.push(problem_records[r]);
        }
        // Sort the problem records by their index (ie. the order
        // they came in. If a later record comes in with the same
        // datalabel as an earlier one, the second index is used).
        pr.sort(function(a,b) {
        var a_ind = problem_records.index[a["metadata"]["datalabel"]];
        var b_ind = problem_records.index[b["metadata"]["datalabel"]];
            if (a_ind<b_ind) {
            return -1;
        }
        if (a_ind>b_ind) {
            return 1;
        }
        return 0;
        });
        var msg = this.formatWarningRecords(pr,"comment");
        var prepend = true;
        this.lightbox.addRecord(msg,prepend);
	// popups have been requested suppressed.
        // $("#lightbox_background, #lightbox_window").show();
	//
	// As alt, the popup can show and then be dismissed after delay
	// in microseconds.
	// $("#lightbox_background, #lightbox_window").show(0).delay(1000).hide(0);
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
            table_record[key[j]] = record[subdict][subkey];
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
            if (error_key != null) {
            var ek = error_key.split("-",2);
            } else {
                ek[1] = null;
            }

            if (record[dk[0]]) {
                var time = record["metadata"]["local_time"];
                var value = record[dk[0]][dk[1]];
                if (ek[1] != null) {
                    var error = record[ek[0]][ek[1]];
                } else {
                    var error = 0.0
                }

                var series;
                if (dk[0]=="iq") {
                    // var wfs = record["metadata"]["wfs"];
                    var wb = record["metadata"]["waveband"];
                    // var oi_bands = ["U","B","V","R","I","Y"];
                    // var pw_bands = ["J","H","K","L","M","N"];
                    //if (wfs=="OIWFS") {
                        //if (oi_bands.indexOf(wb)!=-1) {
                        //    series = wb;
                        //} else {
                        //    series = wb+"(o)";
                        //}
                    //} else if (wfs=="PWFS1" || wfs=="PWFS2") {
                        //if (pw_bands.indexOf(wb)!=-1) {
                        //    series = wb;
                        //} else {
                        //    series = wb+"(p)";
                        //}
                    //} else {
                        //series = wb;
                    //}
                    series = wb;
                } else if (dk[0]=="bg") {
                    series = record["metadata"]["waveband"];
                } else {
                    series = dk[0];
                }

                if (value!=undefined && error!=undefined) {
                    plot_record["series"] = series;
                    plot_record["date"] = time;
                    plot_record["data"] = value;
                    plot_record["error"] = error;
                    plot_record["key"] = record["metadata"]["datalabel"];
                    plot_record["image_number"] =
                    record["metadata"]["image_number"];

                    plt_records.push(plot_record);
                }
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
            if (record[k[0]][k[1]]) {
                if (k[1]=="airmass") {
                tooltip_record["message"] = "AM " +
                                       record[k[0]][k[1]].toFixed(2);
                } else if (k[1]=="ellipticity_str") {
                    tooltip_record["message"] = "Ellip "+record[k[0]][k[1]];
                } else if (k[1]=="waveband") {
                    tooltip_record["message"] = record[k[0]][k[1]] +
                                            "-band";
                } else if (k[1]=="requested_str"){
                    tooltip_record["message"] = "Requested " +
                                            record[k[0]][k[1]];
                } else {
                    tooltip_record["message"] = record[k[0]][k[1]];
                }
            } else {
                tooltip_record["message"] = null;
            }
            } else {
                tooltip_record["message"] = "No data";
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
            var problem = '<span class="msg_problem_icon"></span>';
            var warning = '<span class="msg_warn_icon"></span>';
            for (var j in subdicts) {
            if (record[subdicts[j]]) {
                var msg_array = record[subdicts[j]][key];
                for (var m in msg_array) {
                if (has_msg) {
                    message += ", ";
                }
                if ((msg_array[m].indexOf("ellipticity")!=-1) ||
                    (msg_array[m].indexOf("Single source")!=-1) ||
                    (msg_array[m].indexOf("estimated AO seeing")!=-1) ||
                    (record["metadata"]["types"].indexOf("SPECT")!=-1 )) {
                    message += '<span class="outer">'+
                           warning+"&nbsp;&nbsp;"+
                           '<span class="warning">'+
                           subdicts[j].toUpperCase()+":</span> "+
                           msg_array[m]+
                           '</span>';
                } else {
                    message += '<span class="outer">'+
                           problem+"&nbsp;&nbsp;"+
                           '<span class="problem">'+
                           subdicts[j].toUpperCase()+":</span> "+
                           msg_array[m]+
                           '</span>';
                }
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

    formatWarningRecords: function(records, key) {
        var return_single = false;
        if (!(records instanceof Array)) {
            records = [records];
            return_single = true;
        }

        var msg_records = [];
        for (var i in records) {
            var record = records[i];
            var message = "<h3>Image " +
                          record["metadata"]["image_number"] + ", " +
                      '<span class="datalabel">' +
                          record["metadata"]["datalabel"] +
                          "</span>:</h3><p>";

            var has_msg = false;
            var subdicts = ["iq", "cc", "bg"];
            for (var j in subdicts) {
            var metric = subdicts[j];
            if (record[metric]) {
                var msg_array = record[metric][key];
                if (msg_array.length>0) {

                for (var mi in msg_array) {
                    if (metric=="iq") {
                    if ((msg_array[mi].indexOf("ellipticity")!=-1) ||
                       (record["metadata"]["types"].indexOf("SPECT")!=-1))
                    {
                        continue;
                    } else {
                        message += "<span class=label>"+
                               msg_array[mi]+"</span><br>";
                        message += "Delivered IQ: "+
                               record["iq"]["band_str"] +
                               "<br>";
                        message += "Requested IQ: "+
                               record["iq"]["requested_str"]+
                               "<br>";
                        has_msg = true;
                    }
                    } else if (metric=="cc") {
                    message += "<span class=label>"+
                               msg_array[mi]+"</span><br>";
                    message += "Delivered CC: "+
                               record["cc"]["band_str"] + "<br>";
                    message += "Requested CC: "+
                               record["cc"]["requested_str"] +
                               "<br>";
                    has_msg = true;
                    } else if (metric=="bg") {
                    message += "<span class=label>"+
                               msg_array[mi]+"</span><br>";
                    message += "Delivered BG: "+
                               record["bg"]["band_str"] + "<br>";
                    message += "Requested BG: "+
                               record["bg"]["requested_str"] +
                               "<br>";
                    has_msg = true;
                    }
                }
                }
            }
            }
            if (!has_msg) {
            message += "(none)";
            }

            message +="</p><hr>";

            // Form a key for the message, so that it can be
            // overwritten when a new message comes in for the
            // same observation.  Note that the key cannot
            // just be the datalabel because it is used as an id,
            // and the datalabel is already the id for the table rows
            var msgkey = "warn_"+record["metadata"]["datalabel"];
            msg_records.push({"message":message,
                      "key":msgkey});
        }
        if (return_single) {
            return msg_records[0];
        } else {
            return msg_records;
        }
    }, // end formatWarningRecords

}; // end prototype
