/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */


/**
 *  
 * @param element
 * @param id
 */
function SpectrumViewer(element, id) {
  "use strict";
  
  this.element = element;
  this.id = id;
  
} // end SpectrumViewer


SpectrumViewer.prototype = {
  
  constructor: SpectrumViewer,

  load: function () {
    "use strict";
    var sViewer = this;
  
    $.ajax(
      {
        type: "GET",
        url: "/rqsite.json",
        success: function (data) {
          sViewer.site = data.local_site;
          sViewer.tzname = data.tzname;
          sViewer.utc_offset = data.utc_offset;
      
          // Translate utc_now into a JS data
          var ymd_hms = data.utc_now.split(" "),
            ymd = ymd_hms[0],
            hms = ymd_hms[1];
       
          var year_month_day = ymd.split("-"),
            year = year_month_day[0],
            month = year_month_day[1],
            day  = year_month_day[2];
       
          var hour_minute_second = hms.split(":"),
            hour = hour_minute_second[0],
            minute = hour_minute_second[1],
            second = hour_minute_second[2],
            fraction = (parseFloat(second) - parseInt(second, 10)) * 1000;
      
          second = parseInt(second, 10);
          
          var localDate = 
            new Date(Date.UTC(year, month-1, day, hour, minute, second, fraction));
      
          // Add in the UT offset
          sViewer.server_now = 
            Date(localDate.setHours(localDate.getHours() + sViewer.utc_offset));
       
          // Keep track of the difference between local time and server time
          var localTimezone = localDate.getTimezoneOffset() / 60;
          sViewer.tz_offset = sViewer.utc_offset - localTimezone;
          sViewer.init();
        }, // end success
        error: function () {
          sViewer.site = undefined;
          sViewer.tzname = "LT";
          sViewer.server_now = new Date();
          sViewer.utc_offset = sViewer.server_now.getTimezoneOffset() / 60;
          sViewer.tz_offset = 0;
          sViewer.init(); 
        } // end error
      }); // end ajax
    }, // end load
  
  
  
}