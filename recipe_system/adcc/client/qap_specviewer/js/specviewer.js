/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */
const specViewerJsonName = "/specqueue.json";
const notAvailableYet = '<span class="not-available"> (Not available yet) </span>';
const noData = '<span class="no-data"> No Data </span>';

/**
 * Options to be used by the plots
 */
const plotOptions = {

  axesDefaults: {
    alignTicks: true,
  },

  seriesDefaults: {
    lineWidth: 1,
    markerOptions: {
      size: 1
    },
    shadow: false,
    renderer: $.jqplot.LineRenderer,
  },

  series: [{
      color: 'rgba(46, 134, 193, 1.0)',
      label: 'Intensity',
      shadow: false,
    },
    {
      color: 'rgba(211, 96, 0, 0.2)',
      label: 'Standard Deviation',
      shadow: false,
    },
  ],

  grid: {
    background: 'white',
    drawBorder: false,
    gridLineColor: '#F2F3F4',
    shadow: false,
  },

  legend: {
    show: true,
    location: 'nw'
  },

  cursor: {
    constrainOutsideZoom: false,
    looseZoom: true,
    show: true,
    showTooltip: true,
    showTooltipOutsideZoom: true,
    useAxesFormatters: true,
    zoom: true,
  },

};

/**
 * Adds a count down in the footer to let user know when the server was queried.
 * @param {object} sViewer - An initialized version of SpecViewer.
 */
function addCountDown(sViewer) {

  $(`.footer`).width( $(`#${sViewer.id}`).width() );
  $(`.footer`).append(`<span class="d-table-cell tar countdown"> </span>`);

  $(`.countdown`).append(`<span class="dot"></span>`);
  $(`.countdown`).append(`<span class="title">Querying server in </span>`);
  $(`.countdown`).append(`<span class="number" title=""> </span>`);

  function updateCountdown() {

    if (sViewer.countdown >= 0) {
      $(`.countdown .number`).html(`${sViewer.countdown} s`);
      $(`.countdown .title, .countdown .number, .dot`).addClass(`online`);
      $(`.countdown .title, .countdown .number, .dot`).removeClass(`offline`);
      sViewer.countdown--;
    } else {
      $(`.countdown .title, .countdown .number, .dot`).removeClass(`online`);
      $(`.countdown .title, .countdown .number, .dot`).addClass(`offline`);
    } // end if/else

    $(`.online`).prop('title', "Server is responding");
    $(`.offline`).prop('title', "No or empty response from server");

  } // end updateCountdown
  setInterval(updateCountdown, 1000);
}  // end addCountDown

/**
 * Add the settings button and the hidden form to the webpage.
 * @param {object} sViewer - An initialized version of SpecViewer.
 */
function addSettings(sViewer) {
  'use strict';

  // Add the settings button
  $( `span.settings` ).html(`
    <button id="settingsBtn" class="ui-button ui-widget ui-corner-all settings" title="Settings">
      <img src="/qapspec/images/settings.png"></img>
    </button>
  `);

  // Add the settings form
  $( `div.settings` ).html(`
    <div class="settings-content">

      <h3> Settings </h3>

      <hr>

      <form>

          <label for="queryFreq">Query frequency:</label>
          <input type="number" id="queryFreq" name="queryFreq" min="1" max="60" value=${sViewer.countdown}>
          <label for="queryFreq">seconds</label>
          <br><br>
          <input type="checkbox" name="flashOnRefresh" id="flashOnRefresh" name="flashOnRefresh" value="flash">
          <label for="flashOnRefresh"> Flash file info on refresh </label>

      </form>

      <div class='w-100 d-table'>
        <div class='d-table-cell tar'>
          <button type="button" id="okBtn">Ok</button>
          <button type="button" id="cancelBtn">Cancel</button>
        </div>
      </div>

    </div>
  `);

  let settingsModal = $( `div.settings` );
  let queryFreq = $( "#queryFreq" );

  function showSettings () {
    console.log("Clicked on settings button.");
    settingsModal.css("display", "block");
  };

  function saveSettings () {
    sViewer.delay = queryFreq.val() * 1000;
    sViewer.countdown = queryFreq.val();
    sViewer.gjs.delay = sViewer.delay;
    settingsModal.css("display", "none");
    console.log(` Setting countdown to ${sViewer.delay / 1000} seconds`);
  };

  function cancel() {
    settingsModal.css("display", "none");
    queryFreq.val(sViewer.delay / 1000);
  };

  // Add functionality to buttons
  $( '#settingsBtn' ).unbind("click").on( "click", showSettings );
  $( '#okBtn' ).unbind("click").on( "click", saveSettings );
  $( '#cancelBtn' ).unbind("click").on( "click", cancel );
  $( '.close' ).unbind("click").on( "click", cancel );

  // Close settings if clicked outside
  window.onclick = function(event) {
    if (event.target.id == "settings") {
      cancel();
    }
  }

}

/**
 * Makes sure that every incoming aperture is unique.
 *
 * @param {array} apertures - List of incoming apertures.
 */
function assureUniqueApertures(apertures) {

  let uniqueIds = [];
  let uniqueApertures = [];

  apertures.map(
    function(ap) {
      if (!uniqueIds.includes(ap.id)) {
        uniqueIds.push(ap.id);
        uniqueApertures.push(ap);
      } // end if
    } // end <anonymous>
  ); // end apertures.map()

  return uniqueApertures;

} // end assureUniqueApertures

/**
 * Returns the Aperture Info div element using a template.
 *
 * @type {Object} aperture Aperture definition containing its center, the
 * relative lower limit, the relative upper limit and the dispersion.
 */
function getApertureInfo(aperture) {

  return `
    <div class="apertureInfo">
      <span>
          <span class="app-info-field" title="Aperture information from Latest Frame"> <b>Aperture definition:</b> </span>
          <span class="app-info-field" title="Aperture center"> ${aperture.center} px </span> (
          <span class="app-info-field" title="Lower aperture limit"> ${aperture.lower} px </span>,
          <span class="app-info-field" title="Upper aperture limit"> ${aperture.upper} px </span>)
      </span>
      <span style="padding-left: 10%">
        <b>Dispersion:</b> ${Math.round(aperture.dispersion * 1000) / 1000} nm/px
      </span>
    </div>
    `;

}

/**
 * Get element value inside `list` that is the nearest to the `target` value.
 * @param  {number} target - Target value
 * @param  {array} list - List containg numbers.
 * @return {number} - Nearest element.
 */
function getNearest(target, list) {

  let nearest = list.reduce(
    function(prev, curr) {
      return (Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev);
    }
  );

  return nearest;

}

/**
 * Returns the index of the element inside `list` that is the nearest tho the
 *  `target` value.
 * @param  {number} target - Target value
 * @param  {array} list - List containg numbers.
 * @return {number} - Index of the nearest element.
 */
function getNearestIndex(target, list) {

  let nearest = getNearest(target, list);

  function getIndex(value) {
    return value === nearest;
  }

  return list.findIndex(getIndex);

}


/**
 * Returns the last frame info div element using a template.
 *
 * @param filename {string} name of the filename.
 * @param programId {string} program Id code.
 */
function getFrameInfo(filename, programId) {

  if (filename === '') {
    filename = notAvailableYet;
  }

  if (programId === '') {
    programId = notAvailableYet;
  }

  return `
  <div class="d-table w-100">
    <p class="d-table-cell">
      Latest frame - ${filename} - ${programId}
    </p>
    <div class="d-table-cell tar">
      <button class="ui-button ui-widget ui-corner-all" title="Reset zoom">
        <img class="zoom-reset" src="/qapspec/images/zoom_reset_48px.png"></img>
      </button>
    </div>
  </div>`;
}


/**
 * Returns the stack frame info div element using a template.
 *
 * @param filename {string} name of the filename.
 * @param programId {string} program Id code.
 */
function getStackInfo(filename, programId) {

  if (filename === '') {
    filename = '<span style="color: #aaa"> (Not available yet) </span>';
  }

  if (programId === '') {
    programId = '<span style="color: #aaa"> (Not available yet) </span>';
  }

  return `
    <div class="d-table w-100">
      <p class="d-table-cell">
        Stack frame - ${filename} - ${programId}
      </p>
      <div class="d-table-cell tar">
        <button class="ui-button ui-widget ui-corner-all" title="Reset zoom">
          <img class="zoom-reset" src="/qapspec/images/zoom_reset_48px.png"></img>
        </button>
      </div>
    </div>
  `;
}

/**
 * Convert input units to be used as label to the x-axis.
 * @param {string} units
 */
function getWavelengthUnits(units) {

  if (units) {
    units = $.trim(units);
    units = units.toLowerCase();

    if (["um", "micrometer", "micrometers"].includes(units)) {
      return "Wavelength [um]";
    }

    if (["nm", "nanometer", "nanometers"].includes(units)) {
      return "Wavelength [nm]";
    }

    if (["a", "angsrom", "angsroms"].includes(units)) {
      return "Wavelength [\u212B]";  // escaped Angstrom symbol
    }

  } else {
    return "Wavelength";
  }

}

/**
 * Gets the nearest aperture id value inside `listOfApertures` and verify
 *  if the absolute difference between this nearest value and the expected
 *  `aperture` value is smaller than the tolerance value.
 *
 * @param  {number}  aperture - Input aperture id value.
 * @param  {number}  pixelScale - Binned pixel scale in arcseconds per pixel.
 * @param  {array}  listOfApertures - List of apertures ids.
 * @return {boolean} - Returns True is absolute difference between aperture and
 *   its nearest value is smaller than the tolerance.
 */
function isInApertureList(aperture, pixelScale, listOfApertures) {

  let tolerance = 1.0;  // arcseconds
  let nearest = getNearest(aperture, listOfApertures);
  let nearestIndex = getNearestIndex(aperture, listOfApertures);

  let matches = (Math.abs(aperture - nearest) * pixelScale <= tolerance);

  return matches;

}

/**
 * Main component for SpecViewer.
 *
 * @param {object} parentElement - element that will hold SpecViewer.
 * @param {string} id - name of the ID of the SpecViewer div container.
 * @param {int} queryFreq - refresh delay in seconds
 */
class SpecViewer {

  constructor(parentElement, id, queryFreq) {
    'use strict';

    // Creating empty object
    this.parentElement = parentElement;
    this.id = id;

    // Placeholders for different elements
    this.singlePlots = [];
    this.stackPlots = [];

    this.aperturesId = [];
    this.dataLabel = null;  // Last Frame Data Label
    this.groupId = null;
    this.stackSize = 0;
    this.timestamp = 0;

    // Create empty page
    this.parentElement.html(`
        <div id="${id}">
          <div class="loading">
            Waiting data from server ...
          </div>
          <ul></ul>
        </div>
      `);

    // Call function to enable the tabs
    $(`#${id}`).tabs();

    // Add countdown
    this.delay = queryFreq * 1000;
    this.countdown = queryFreq;

    addCountDown(this);
    addSettings(this);

    // Placeholder for adcc command pump
    this.gjs = null;

  } // end constructor

  /**
   * Flashes the file information if new data arrives.
   *
   * @param {string} type
   */
  flashInfo(type) {

    if ( $( `#flashOnRefresh` ).is(':checked') ) {

      let defaultBackgroundColor = $( `div.info.${type}` ).css("background-color");

      $( `div.info.${type}` ).animate({
        backgroundColor: "#ACD5FF",
      }, 500 );

      $( `div.info.${type}` ).animate({
        backgroundColor: defaultBackgroundColor,
      }, 1500 );

    }

  }

  /**
   * Query server for JSON file and start to populate page.
   * This function is the registered callback on the command pump.
   *
   * @param {object} jsonData
   */
  loadData(jsonData) {
      'use restrict';

      this.now = Date(Date.now());

      // Restart countdown
      this.countdown = this.delay / 1000;

      // Clear console
      console.clear();

      // Remove loading
      $('.loading').remove();

      console.log(`\nReceived new JSON data list (${jsonData.length} elements) on\n ${this.now.toString()}`);

      // Process incoming data
      for (let i = 0; i < jsonData.length; i++) {

        let dataLabel = jsonData[i].data_label;
        let isStack = jsonData[i].is_stack;
        let newData = false;
        let offset = jsonData[i].offset;
        let stackSize = jsonData[i].stack_size;
        let type = (isStack) ? "stack":"single";

        let jsonElement = jsonData[i];

        jsonElement.apertures = assureUniqueApertures(jsonElement.apertures);

        if (jsonElement.group_id !== this.groupId) {

          console.log(`- NEW group Id: ${jsonElement.group_id} - ${type} data`);
          this.groupId = jsonElement.group_id;

          // Clear navigation tabs and relevant lists
          this.singlePlots = [];
          this.stackPlots = [];
          this.stackSize = 0;

          // Clear tab contents
          $('.tabcontent').remove();

          // Clear navigation tab
          $(`#${this.id} ul`).empty();
          $(`#${this.id}`).tabs('refresh');

          // Add tab content for every aperture
          this.aperturesId = jsonElement.apertures.map(
            function(a) { return a.id; });

          this.aperturesId.sort();

          for (let i = 0; i < this.aperturesId.length; i++) {
            this.newTabContent(jsonElement.apertures[i]);
          }

          // Update relevant values and plots
          if (isStack) {
            $('.footer .status').html(`${this.now.toString()}<br />Received new stack data with ${jsonElement.apertures.length} aperture(s)`);
            this.stackSize = jsonElement.stack_size;
          } else {
            $('.footer .status').html(`${this.now.toString()}<br />Received new data with ${jsonElement.apertures.length} aperture(s)`);
            this.dataLabel = jsonElement.data_label;
          }

          this.timestamp = jsonElement.timestamp;
          this.updatePlotArea(jsonElement, type);
          this.updateNavigationTab();
          this.flashInfo(type);

        } else {

          console.log(`- SAME group Id: ${this.groupId} - ${type} data`);

          // Check if new incoming data have apertures not in aperture id
          for (let i = 0; i < jsonElement.apertures.length; i++) {

            let ap = jsonElement.apertures[i];
            let ps = jsonElement.pixel_scale;

            if ( !isInApertureList(ap.id, ps, this.aperturesId) ) {
              console.log( `Found new aperture: ${ap.id}` );
              this.newTabContent( ap );
              this.aperturesId.push( ap.id );
            }

          }

          if (isStack) {
            if (stackSize == this.stackSize && this.timestamp >= jsonElement.timestamp) {
              console.log(`- OLD stack data with ${stackSize} frames (${jsonElement.apertures.length} apertures)`);
              $('.footer .status').html(`${this.now.toString()}<br />No new data from last request.`);
            } else {
              console.log(`- NEW stack data with ${stackSize} frames (${jsonElement.apertures.length} apertures)`);
              $('.footer .status').html(`${this.now.toString()}<br />Received new stack data with ${jsonElement.apertures.length} aperture(s)`);
              this.timestamp = jsonElement.timestamp;
              this.stackSize = stackSize;
              this.updatePlotArea(jsonElement, type);
              this.updateNavigationTab();
              this.flashInfo(type);
            }
          } else {
            if (this.dataLabel === jsonElement.data_label && this.timestamp >= jsonElement.timestamp) {
              console.log(`- OLD frame data: ${this.dataLabel} (${jsonElement.apertures.length} apertures)`);
              $('.footer .status').html(`${this.now.toString()}<br />No new data from last request.`);
            } else {
              console.log(`- NEW frame data: ${jsonElement.data_label} (${jsonElement.apertures.length} apertures)`);
              $('.footer .status').html(`${this.now.toString()}<br />Received new data with ${jsonElement.apertures.length} aperture(s)`);
              this.timestamp = jsonElement.timestamp;
              this.dataLabel = jsonElement.data_label;
              this.updatePlotArea(jsonElement, type);
              this.updateNavigationTab();
              this.flashInfo(type);
            }
          }
        }
      }
      this.updateUiBehavior();

    } // end load

    /**
     * Add tab content for incoming data.
     * @type {object} ap - Incoming aperture data
     */
    newTabContent(ap) {

      $(`#${this.id}`).append(
        `<div id="aperture${ap.id}" class="tabcontent">
          <div class="apertureInfo"> </div>
          <div class="info single"> </div>
          <div class="ui-widget-content resizable single" id="singlePlot${ap.id}-resizable" >
            ${notAvailableYet}
          </div>
          <div class="info stack"> </div>
          <div class="ui-widget-content resizable stack" id="stackPlot${ap.id}-resizable" >
            ${notAvailableYet}
          </div>
        </div>`
      );

      // Add content to the container
      $(`#aperture${ap.id} .apertureInfo`).html( getApertureInfo(ap) );
      $(`#aperture${ap.id} .info.single`).html( getFrameInfo("", "") );
      $(`#aperture${ap.id} .info.stack`).html( getStackInfo("", "") );

  } // end newTabContent

  /**
   * Enable Reset Zoom button for Frame Plots
   * @param {object} p - JQPlot instance
   * @param {number} i - Aperture index
   * @param {string} type - Aperture type (single/stack)
   */
  resetZoom(p, i, type) {

      let sViewer = this;
      let apId = sViewer.aperturesId[i];

      function sleep (miliseconds) {
        return new Promise(resolve => setTimeout(resolve, miliseconds));
      }

      // Unbind click to prevent setting it several times
      $(`#aperture${apId} .info.${type} button`).unbind( "click" );

      // Bind click event
      $(`#aperture${apId} .info.${type} button`).click(
        async function() {
          console.log(`Reset zoom of ${type} plot #${i}.`);
          p.resetZoom();
          sleep(250);
        }
      );
  } //

  /**
   * Starts to query for new data
   */
  start() {

    console.log("Starting SpecViewer");

    // Make an AJAX request to the server for the current
    // time and the server site information.
    // The callback for this request will call the init function
    var sv = this;

    $.ajax({
      type: "GET",
      url: "/rqsite.json",
      success: function success (data) {
        sv.site = data.local_site;
        sv.timestamp = data.unxtime;
      }, // end success
      error: function error () {
        sv.site = undefined;
        sv.tzname = "LT";
        sv.timestamp = new Date();
      } // end error
    }); // end ajax

    this.gjs = new GJSCommandPipe();
    this.gjs.registerCallback("specjson", function(msg) {
      sv.loadData(msg);
    });
    this.gjs.startPump(sv.timestamp, "specjson");
    this.gjs.delay = this.delay;

  } // end start

  /**
   * Updates navigation tabs based on tab contents and the number of aperture
   * ids registered inside SpecViewer
   */
  updateNavigationTab() {

    // Add navigation tab container
    let navTabContainer = $(`#${this.id} ul`);

    // Empty tab containers
    navTabContainer.empty();

    // Sort appertures to make our life easier
    this.aperturesId.sort();

    // Create buttons and add them to the navigation tab
    for (let i = 0; i < this.aperturesId.length; i++) {
      navTabContainer.append(`
        <li><a href="#aperture${this.aperturesId[i]}"> Aperture ${this.aperturesId[i]}
        </a></li>`);
    }

    // Save active tab index to recover after
    let activeTabIndex = $(`#${this.id}`).tabs('option', 'active');

    /// Refresh tabs to update them
    $(`#${this.id}`).tabs('refresh');

    // Activate first tab if none is active
    activeTabIndex = activeTabIndex ? activeTabIndex:0;
    $(`#${this.id}`).tabs('option', 'active', activeTabIndex);

} // end updateNavigationTab

  /**
   * Updates the plot area
   * @type {object} data -
   * @type {string} type -
   */
  updatePlotArea(data, type) {

    for (let i = 0; i < this.aperturesId.length; i++) {

      let apertureId = this.aperturesId[i];
      let plotId = `${type}Plot_${apertureId}`;
      let activeTabIndex = $(`#${this.id}`).tabs('option', 'active');

      let inputAperturesId = data.apertures.map(function (a) {return a.id;});
      let apIdx = getNearestIndex(apertureId, inputAperturesId);

      let intensity = data.apertures[apIdx].intensity;
      let stddev = data.apertures[apIdx].stddev;
      let intensityUnits = data.apertures[apIdx].intensity_units;
      let wavelengthUnits = data.apertures[apIdx].wavelength_units;

      let stackTitle = `Aperture ${i + 1} - Stack Frame - Stack size: ${this.stackSize}`;
      let lastTitle = `Aperture ${i+1} - Last Frame - ${this.dataLabel}`;
      let plotTitle = (data.is_stack) ? stackTitle:lastTitle;

      $(`#aperture${apertureId} .apertureInfo`).html(
        getApertureInfo(data.apertures[apIdx])
      );

      $(`#aperture${apertureId} .info.${type}`).html(
        getFrameInfo(data.filename, data.program_id)
      );

      // Create plot area if it does not exist
      if (!$(`#${plotId}`).length) {
        $(`#aperture${apertureId} .resizable.${type}`).html(
          `<div class="plot ${type}" id="${plotId}"> </div>`);

      }

      // Plot instance exists
      if (this[`${type}Plots`][i]) {

        // Existing plotted apperture id exists inside data apertures
        if (isInApertureList(apertureId, data.pixel_scale, inputAperturesId)) {

          console.log('Refresh plots');

          this[`${type}Plots`][i].title.text = plotTitle;
          this[`${type}Plots`][i].series[0].data = intensity;
          this[`${type}Plots`][i].series[1].data = stddev;
          this[`${type}Plots`][i].resetAxesScale();

          // Refresh only on active tab
          if (i === activeTabIndex) {
            this[`${type}Plots`][i].replot( { resetAxes:true } );
          }

        } else {
          $(`#${plotId}`).html(noData);
        }

      } else {

        // Existing plotted apperture id exists inside data apertures
        if (isInApertureList(apertureId, data.pixel_scale, inputAperturesId)) {

          console.log('Create new plots');

          this[`${type}Plots`][i] = $.jqplot(
            plotId, [intensity, stddev], $.extend(plotOptions, {
              title: plotTitle,
              axes: {
                xaxis: {
                  label: getWavelengthUnits(wavelengthUnits),
                  labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
                },
                yaxis: {
                  label: `Intensity [${intensityUnits}]`,
                  labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
                  tickOptions:{formatString:'%.2e'},
                },
              },
            })
          );

        } else {
          $(`#${plotId}`).html(noData);
        }

      }

    }

  }

  /**
   * Updates the UI components behavior (buttons, plots, etc.)
   */
  updateUiBehavior() {

    // Reference self to use in inner function
    let sViewer = this;

    this.singlePlots.map(function(p, i) {
        sViewer.resetZoom(p, i, 'single');
    });

    this.stackPlots.map(function(p, i) {
        sViewer.resetZoom(p, i, 'stack');
    });

    // Enable on tab change event
    function resizePlotArea(index, type) {
      let apId = sViewer.aperturesId[index];
      let plotInstance = sViewer[`${type}Plots`][index];
      let plotTarget = $(`#${type}Plot_${apId}`);
      let resizableArea = $(`#aperture${apId} .resizable.${type}`);

      plotTarget.width(resizableArea.width() * 0.99);

      // Sometimes this function is activated before plots are defined.
      // if ($(plotTarget).length) {
      //   if (plotInstance) {
      //     plotInstance.replot({ resetAxes: true }
      //     );
      //   }
      // }

      if (plotInstance) {
        plotInstance.replot({ resetAxes: true }
        );
      }

    }

    function onTabChange(event, tab) {
      let newIndex = tab.newTab.index();
      resizePlotArea(newIndex, 'single');
      resizePlotArea(newIndex, 'stack');
    }

    $(`#${this.id}`).tabs({'activate': onTabChange});

    // Resize plot area on window resize stop
    function resizeEnd() {
      let activeTabIndex = $(`#${sViewer.id}`).tabs('option', 'active');
      resizePlotArea(activeTabIndex, 'single');
      resizePlotArea(activeTabIndex, 'stack');
    }

    var doit;
    window.onresize = function() {
      clearTimeout(doit);
      doit = setTimeout(resizeEnd, 500);
    };

    // Enable to resize plot area's height
    $('.resizable').resizable({
      ghost: true,
      resize: function(event, ui) {

        // Restrict horizontal resizing
        ui.size.width = ui.originalSize.width;

        // Resize internal elements
        let id = $(this).children('.plot').attr('id');
        let index = $(`#${sViewer.id}`).tabs('option', 'active');
        let type = (id.includes('stack') ? 'stack':'single');

        $('.plot').height(ui.size.height * 0.95);
        sViewer[`${type}Plots`][index].replot( { resetAxes: true } );

      }
    });

  }

} // end SpecViewer
