/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */
const specViewerJsonName = "/specqueue.json";
const notAvailableYet = '<span class="not-available"> (Not available yet) </span>';
const noData = '<span class="no-data"> No Data </span>';


/**
 * Main component for SpecViewer.
 *
 * @param {object} parentElement - element that will hold SpecViewer.
 * @param {string} id - name of the ID of the SpecViewer div container.
 */
function SpecViewer(parentElement, id, delay) {
  'use strict';

  // Creating empty object
  this.parentElement = parentElement;
  this.id = id;

  // Placeholders for different elements
  this.singlePlots = [];
  this.stackPlots = [];

  this.aperturesCenter = [];
  this.dataLabel = null;  // Last Frame Data Label
  this.groupId = null;
  this.stackSize = 0;

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
  this.delay = delay;
  this.countdown = delay / 1000;
  addCountDown(this);

  // Placeholder for adcc command pump
  this.gjs = null;
  this.start();

}
//this.specPump = new SpectroPump();
//} // end SpecViewer

// Add methods to prototype
SpecViewer.prototype = {

  constructor: SpecViewer,

  /**
   * Add/refresh tab content for incomming data.
   * @type {object} data Incomming JSON data
   */
  newTabContent: function(aperture) {

    $(`#${this.id}`).append(
      `<div id="aperture${aperture.center}" class="tabcontent">
        <div class="apertureInfo"> </div>
        <div class="info single"> </div>
        <div class="ui-widget-content resizable single" id="singlePlot${aperture.center}-resizable" >
          ${notAvailableYet}
        </div>
        <div class="info stack"> </div>
        <div class="ui-widget-content resizable stack" id="stackPlot${aperture.center}-resizable" >
          ${notAvailableYet}
        </div>
      </div>`
    );

    // Add content to the container
    $(`#aperture${aperture.center} .apertureInfo`).html(
      getApertureInfo(aperture)
    );

    $(`#aperture${aperture.center} .info.single`).html(
      getFrameInfo("", "")
    );

    $(`#aperture${aperture.center} .info.stack`).html(
      getStackInfo("", "")
    );

  },

  /**
   * Query server for JSON file and start to populate page.
   * This function is the registered callback on the command pump.
   *
   * @param jsonData {object}
   */
  loadData: function(jsonData) {
    'use restrict';

    let now = Date(Date.now());

    // Restart countdown
    this.countdown = this.delay / 1000;

    // Clear console
    console.clear();

    // Remove loading
    $('.loading').remove();

    console.log(`\nReceived new JSON data list (${jsonData.length} elements) on\n ${now.toString()}`);

    // Process incoming data
    for (let i = 0; i < jsonData.length; i++) {

      let dataLabel = jsonData[i].data_label;
      let isStack = jsonData[i].is_stack;
      let newData = false;
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
        this.aperturesCenter = jsonElement.apertures.map(
          function(a) { return Math.round(a.center); });

        this.aperturesCenter.sort();

        for (let i = 0; i < this.aperturesCenter.length; i++) {
          this.newTabContent(jsonElement.apertures[i]);
        }

        // Update relevant values and plots
        if (isStack) {
          $('.footer .status').html(`Received new stack data with ${jsonElement.apertures.length} aperture(s)`);
          this.stackSize = jsonElement.stack_size;
        } else {
          $('.footer .status').html(`Received new data with ${jsonElement.apertures.length} aperture(s)`);
          this.dataLabel = jsonElement.data_label;
        }

        this.updatePlotArea(jsonElement, type);
        this.updateNavigationTab();

      } else {

        console.log(`- SAME group Id: ${this.groupId} - ${type} data`);

        // Check if new incoming data have apertures not in aperture center
        for (let i = 0; i < jsonElement.apertures.length; i++) {

          let ap = jsonElement.apertures[i];
          let ps = jsonElement.pixel_scale;

          if (!isInApertureList(ap.center, ps, this.aperturesCenter)) {
            console.log( `Found new aperture: ${ap.center}` );
            this.newTabContent( ap );
            this.aperturesCenter.push( Math.round(ap.center) );
          }

        }

        if (isStack) {
          if (stackSize <= this.stackSize) {
            console.log(`- OLD stack data with ${stackSize} frames (${jsonElement.apertures.length} apertures)`);
            $('.footer .status').html(`No new data from last request.`);
          } else {
            console.log(`- NEW stack data with ${stackSize} frames (${jsonElement.apertures.length} apertures)`);
            $('.footer .status').html(`Received new stack data with ${jsonElement.apertures.length} aperture(s)`);
            this.stackSize = stackSize;
            this.updatePlotArea(jsonElement, type);
            this.updateNavigationTab();
          }
        } else {
          if (this.dataLabel === jsonElement.data_label) {
            console.log(`- OLD frame data: ${this.dataLabel} (${jsonElement.apertures.length} apertures)`);
            $('.footer .status').html(`No new data from last request.`);
          } else {
            console.log(`- NEW frame data: ${jsonElement.data_label} (${jsonElement.apertures.length} apertures)`);
            $('.footer .status').html(`Received new data with ${jsonElement.apertures.length} aperture(s)`);
            this.dataLabel = jsonElement.data_label;
            this.updatePlotArea(jsonElement, type);
            this.updateNavigationTab();
          }
        }
      }
    }
    this.updateUiBehavior();

  }, // end load


  /**
   * [description]
   * @return {[type]} [description]
   */
  start: function() {

    console.log("Starting SpecViewer");

    // Make an AJAX request to the server for the current
    // time and the server site information.
    // The callback for this request will call the init function
    var sv = this;

    $.ajax({
      type: "GET",
      url: "/rqsite.json",
      success: function(data) {
        sv.site = data.local_site;
        sv.timestamp = data.unxtime;
      }, // end success
      error: function() {
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

  }, // end start

  /**
   * Updates the UI components behavior (buttons, plots, etc.)
   */
  updateUiBehavior: function() {

    // Reference self to use in inner function
    let sViewer = this;

    // Enable Reset Zoom button for Frame Plots
    function resetZoom(p, i, type) {
      let apertureCenter = sViewer.aperturesCenter[i];
      $(`#aperture${apertureCenter} .info.${type} button`).click( function() {
        console.log(`Reset zoom of ${type} plot #${i}.`);
        p.resetZoom();
      });
    }

    this.singlePlots.map(function(p, i) {
      resetZoom(p, i, 'single');
    });

    this.stackPlots.map(function(p, i) {
      resetZoom(p, i, 'stack');
    });

    // Enable on tab change event
    function resizePlotArea(index, type) {
      let apCenter = sViewer.aperturesCenter[index];
      let plotInstance = sViewer[`${type}Plots`][index];
      let plotTarget = $(`#${type}Plot_${apCenter}`);
      let resizableArea = $(`#aperture${apCenter} .resizable.${type}`);

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

  },

  /**
   * Updates navigation tabs based on tab contents and the number of aperture
   * centers registered inside SpecViewer
   */
  updateNavigationTab: function() {

    // Add navigation tab container
    let navTabContainer = $(`#${this.id} ul`);

    // Empty tab containers
    navTabContainer.empty();

    // Sort appertures to make our life easier
    this.aperturesCenter.sort();

    // Create buttons and add them to the navigation tab
    for (let i = 0; i < this.aperturesCenter.length; i++) {
      navTabContainer.append(`
        <li><a href="#aperture${this.aperturesCenter[i]}"> Aperture ${i + 1}
        </a></li>`);
    }

    // Save active tab index to recover after
    let activeTabIndex = $(`#${this.id}`).tabs('option', 'active');

    /// Refresh tabs to update them
    $(`#${this.id}`).tabs('refresh');

    // Activate first tab if none is active
    activeTabIndex = activeTabIndex ? activeTabIndex:0;
    $(`#${this.id}`).tabs('option', 'active', activeTabIndex);

  },

  updatePlotArea: function(data, type) {

    for (let i = 0; i < this.aperturesCenter.length; i++) {

      let apertureCenter = this.aperturesCenter[i];
      let plotId = `${type}Plot_${apertureCenter}`;
      let activeTabIndex = $(`#${this.id}`).tabs('option', 'active');

      let inputAperturesCenter = data.apertures.map(function (a) {return a.center;});
      let apertureIndex = getNearestIndex(apertureCenter, inputAperturesCenter);

      let intensity = data.apertures[apertureIndex].intensity;
      let stddev = data.apertures[apertureIndex].stddev;
      let units = data.apertures[apertureIndex].wavelength_units;

      let stackTitle = `Aperture ${i + 1} - Stack Frame - Stack size: ${this.stackSize}`;
      let lastTitle = `Aperture ${i+1} - Last Frame - ${this.dataLabel}`;
      let plotTitle = (data.is_stack) ? stackTitle:lastTitle;

      $(`#aperture${apertureCenter} .info.${type}`).html(
        getFrameInfo(data.filename, data.program_id)
      );

      // Create plot area if it does not exist
      if (!$(`#${plotId}`).length) {
        $(`#aperture${apertureCenter} .resizable.${type}`).html(
          `<div class="plot ${type}" id="${plotId}"> </div>`);

      }

      // Plot instance exists
      if (this[`${type}Plots`][i]) {

        // Existing plotted apperture center exists inside data apertures
        if (isInApertureList(apertureCenter, data.pixel_scale, inputAperturesCenter)) {

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

        // Existing plotted apperture center exists inside data apertures
        if (isInApertureList(apertureCenter, data.pixel_scale, inputAperturesCenter)) {

          console.log('Create new plots');

          this[`${type}Plots`][i] = $.jqplot(
            plotId, [intensity, stddev], $.extend(plotOptions, {
              title: plotTitle,
              axes: {
                xaxis: {
                  label: getWavelengthUnits(units),
                  labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
                },
                yaxis: {
                  label: "Intensity [e\u207B]", // escaped superscript minus
                  labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
                },
              },
            })
          );

        } else {
          $(`#${plotId}`).html(noData);
        }

      }

    }

  },

}; // end prototype


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
        <b>Aperture definition:</b>
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
 * Adds a count down in the footer to let user know when the server was queried.
 * @param {object} sViewer - An initialized version of SpecViewer.
 */
function addCountDown(sViewer) {

  $(`.footer`).width( $(`#${sViewer.id}`).width() );
  $(`.footer`).append(`<div class="d-table-cell tar countdown"> </div>`);

  $(`.countdown`).append(`<span class="dot"></span>`);
  $(`.countdown`).append(`<spam class="title">Querying server in </spam>`);
  $(`.countdown`).append(`<spam class="number" title=""> </spam>`);

  function updateCountdown() {

    if (sViewer.countdown >= 0) {
      $(`.countdown .number`).html(`${sViewer.countdown} s`);
      $(`.countdown .title, .countdown .number, .dot`).addClass(`online`);
      $(`.countdown .title, .countdown .number, .dot`).removeClass(`offline`);
      sViewer.countdown--;
    } else {
      $(`.countdown .title, .countdown .number, .dot`).removeClass(`online`);
      $(`.countdown .title, .countdown .number, .dot`).addClass(`offline`);
    }

    $(`.online`).prop('title', "Server is responding");
    $(`.offline`).prop('title', "No or empty response from server");

  }

  setInterval(updateCountdown, 1000);

}

/**
 * Makes sure that every incoming aperture is unique.
 *
 * @param {array} apertures - List of incoming apertures.
 */
function assureUniqueApertures(apertures) {

  let uniqueCenters = [];
  let uniqueApertures = [];

  apertures.map(
    function(ap) {
      if (!uniqueCenters.includes(ap.center)) {

        uniqueCenters.push(ap.center);
        uniqueApertures.push(ap);
      }
    }
  );

  return uniqueApertures;

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
        <img class="zoom-reset" src="/qlook/images/zoom_reset_48px.png"></img>
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
          <img class="zoom-reset" src="/qlook/images/zoom_reset_48px.png"></img>
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
 * Gets the nearest aperture center value inside `listOfApertures` and verify
 *  if the absolute difference between this nearest value and the expected
 *  `aperture` value is smaller than the tolerance value.
 *
 * @param  {number}  aperture - Input aperture center value.
 * @param  {number}  pixelScale - Binned pixel scale in arcseconds per pixel.
 * @param  {array}  listOfApertures - List of apertures centers.
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
 * Options to be used by the plots
 */
plotOptions = {

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
      color: '#1f77b4',
      label: 'Intensity',
    },
    {
      color: 'rgba(255, 127, 14, 0.2)',
      label: 'Standard Deviation'
    },
  ],

  grid: {
    background: 'white',
    drawBorder: false,
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
