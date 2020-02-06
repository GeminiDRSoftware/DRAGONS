/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */
const specViewerJsonName = "/specqueue.json";
const notAvailableYet = '<span class="not-available"> (Not available yet) </span>';


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
  this.activeTab = null;
  this.framePlots = [];
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
   * Add plots to the existing HTML elements.
   */
  addPlots: function(parentId, data) {
    'use restrict';

    let sViewer = this;

    var intensity = null;
    var stddev = null;

    let framePlots = [];
    let stackPlots = [];

    for (var i = 0; i < data.apertures.length; i++) {

      // Adding plot for frame
      intensity = buildSeries(
        data.apertures[i].wavelength, data.apertures[i].intensity);

      stddev = buildSeries(
        data.apertures[i].wavelength, data.apertures[i].stddev);

      framePlots[i] = $.jqplot(
        `framePlot${i}`, [intensity, stddev], $.extend(plotOptions, {
          title: `Aperture ${i} - Last Frame`,
        }));

      // Adding plots for stack
      intensity = buildSeries(
        data.stackApertures[i].wavelength, data.stackApertures[i].intensity);

      stddev = buildSeries(
        data.stackApertures[i].wavelength, data.stackApertures[i].stddev);

      stackPlots[i] = $.jqplot(
        `stackPlot${i}`, [intensity, stddev], $.extend(plotOptions, {
          title: `Aperture ${i} - Stack Frame`,
        }));

    }

    // Save plots references in the object itself
    this.framePlots = framePlots;
    this.stackPlots = stackPlots;

    // Add select tab handler
    var selectTab = function(e, tab) {
      console.log("Selected tab ", tab.newTab.index());
      sViewer.resizeFramePlots(tab.newTab.index());
      sViewer.resizeStackPlots(tab.newTab.index());
    };

    // Call function to activate the tabs
    // $(`#${parentId}`).tabs('refresh');
    // $(`#${parentId}`).tabs('option', 'active', 0);
    // $(`#${parentId}`).tabs({
    //   'activate': selectTab
    // });

    // Allow plot area to be resized
    $('.ui-widget-content.resizable:has(.framePlot)').map(
      function onResizeFrameStop(index, element) {

        $(element).resizable({
          delay: 20,
          helper: 'ui-resizable-helper'
        });

        $(element).bind('resizestop', function resizeFramePlot(event, ui) {
          $(`framePlot${index}`).height($(element).height() * 0.96);
          $(`framePlot${index}`).width($(element).width() * 0.96);

          framePlots[index].replot({
            resetAxes: true
          });
        });

      });

    $('.ui-widget-content.resizable:has(.stackPlot)').map(
      function onResizeStackStop(index, element) {

        $(element).resizable({
          delay: 20,
          helper: 'ui-resizable-helper'
        });

        $(element).bind('resizestop', function resizeStackPlot(event, ui) {
          $(`stackPlot${index}`).height($(element).height() * 0.96);
          $(`stackPlot${index}`).width($(element).width() * 0.96);

          stackPlots[index].replot({
            resetAxes: true
          });
        });

      });

  },

  /**
   * Add/refresh tab content for incomming data.
   * @type {object} data Incomming JSON data
   */
  newTabContent: function(aperture) {

    $(`#${this.id}`).append(
      `<div id="aperture${aperture.center}" class="tabcontent">
        <div class="apertureInfo"> </div>
        <div class="info frame"> </div>
        <div class="ui-widget-content resizable frame" id="framePlot${aperture.center}-resizable" >
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

    $(`#aperture${aperture.center} .info.frame`).html(
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

    console.log(`\nReceived new JSON data list on\n ${now.toString()}`);

    // Process incoming data
    for (let i = 0; i < jsonData.length; i++) {

      let dataLabel = jsonData[i].data_label;
      let isStack = jsonData[i].is_stack;
      let newData = false;
      let stackSize = jsonData[i].stack_size;

      let jsonElement = jsonData[i];

      if (jsonElement.group_id !== this.groupId) {

        console.log(`- NEW data with group ID: ${jsonElement.group_id}`);
        this.groupId = jsonElement.group_id;

        // Clear navigation tabs and relevant lists
        this.framePlots = [];
        this.stackPlots = [];
        this.stackSize = 0;

        // Clear tab contents
        $('.tabcontent').remove();

        // Add tab content for every aperture
        this.aperturesCenter = jsonElement.apertures.map(
          function(a) { return a.center; });

        this.aperturesCenter.sort();

        for (let i = 0; i < this.aperturesCenter.length; i++) {
          this.newTabContent(jsonElement.apertures[i]);
        }

        if (isStack) {
          this.updateStackArea(jsonElement);
        } else {
          this.updateFrameArea(jsonElement);
        }

      } else {

        console.log(`- Data from SAME group Id: ${this.groupId}`);

        for (let i = 0; i < jsonElement.apertures.length; i++) {
          let ap = jsonElement.apertures[i];
          let ps = jsonElement.pixel_scale;

          if (notInApertureCenterList(ap.center, ps, this.aperturesCenter)) {
            this.newTabContent(ap);
          }
        }

        // if (isStack) {
        //   if (stackSize > this.stackSize) {
        //     console.log(`- NEW stack data with ${stackSize}`);
        //     this.updateStackArea(jsonElement);
        //   } else {
        //     console.log(`- OLD stack data with ${stackSize}`);
        //   }
        // } else {
        //   if (this.dataLabel === jsonElement.data_label) {
        //     console.log(`- OLD frame data: ${this.dataLabel}`);
        //   } else {
        //     console.log(`- NEW frame data: ${jsonElement.data_label}`);
        //     this.updateFrameArea(jsonElement);
        //   }
        // }

      }

      this.updateNavigationTab();

    }

    // Update UI behavior
    this.updateUiBehavior();

  }, // end load

  /**
   * Resizes frame plots on different situations, like window resizing or
   * when changing tabs.
   *
   * @param activeTabIndex {number}
   */
  resizeFramePlots: function(activeTabIndex) {

    console.log(`Resizing frame plot ${activeTabIndex}`);

    $(`framePlot${activeTabIndex}`).height(
      $(`framePlot${activeTabIndex}-resizable`).height() * 0.96
    );

    $(`framePlot${activeTabIndex}`).width(
      $(`framePlot${activeTabIndex}-resizable`).width() * 0.96
    );

    this.framePlots[activeTabIndex].replot({
      resetAxes: true
    });

  },

  /**
   * Resizes frame plots on different situations, like window resizing or
   * when changing tabs.
   *
   * @param activeTabIndex {number}
   */
  resizeStackPlots: function(activeTabIndex) {

    console.log(`Resizing stack plot ${activeTabIndex}`);

    $(`stackPlot${activeTabIndex}`).height(
      $(`stackPlot${activeTabIndex}-resizable`).height() * 0.96
    );

    $(`stackPlot${activeTabIndex}`).width(
      $(`stackPlot${activeTabIndex}-resizable`).width() * 0.96
    );

    this.stackPlots[activeTabIndex].replot({
      resetAxes: true
    });

  },

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

    let sViewer = this;

    // Enable Reset Zoom button for Frame Plots
    function resetZoom(p, i, type) {
      let apertureCenter = sViewer.aperturesCenter[i];
      $(`#aperture${apertureCenter} .info.${type} button`).click( function() {
        console.log(`Reset zoom of ${type} plot #${i}.`);
        p.resetZoom();
      });
    }

    this.framePlots.map(function(p, i) {
      resetZoom(p, i, 'frame');
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

      console.log('Aperture center: ', apCenter);

      plotTarget.height(resizableArea.height() * 0.96);
      plotTarget.width(resizableArea.width() * 0.96);

      // Sometimes this function is activated before plots are defined.
      if (plotInstance) {
        plotInstance.replot({ resetAxes: true });
      }

    }

    function onTabChange(event, tab) {
      let newIndex = tab.newTab.index();
      resizePlotArea(newIndex, 'frame');
      resizePlotArea(newIndex, 'stack');
    }

//    $(`#${this.id}`).tabs({'activate': onTabChange});

    // Resize plot area on window resize stop
    function resizeEnd() {
      let activeTabIndex = $(`#${sViewer.id}`).tabs('option', 'active');
      resizePlotArea(activeTabIndex, 'frame');
      resizePlotArea(activeTabIndex, 'stack');
    }

    var doit;
    window.onresize = function() {
      clearTimeout(doit);
      doit = setTimeout(resizeEnd, 500);
    };

  },

  /**
   * [description]
   * @param  {[type]} data [description]
   * @return {[type]}      [description]
   */
  updateFrameArea: function(data) {

    this.dataLabel = data.data_label;
    console.log(`- Handling latest frame data: ${this.dataLabel}`);

    for (let i = 0; i < this.aperturesCenter.length; i++) {

      let apertureCenter = this.aperturesCenter[i];
      let framePlotId = `framePlot_${apertureCenter}`;
      let intensity = data.apertures[i].intensity;
      let stddev = data.apertures[i].stddev;
      let units = data.apertures[i].wavelength_units;
      let activeTabIndex = $(`#${this.id}`).tabs('option', 'active');

      $(`#aperture${apertureCenter} .info.frame`).html(
        getFrameInfo(data.filename, data.program_id)
      );

      // Check if plot containers exist
      if (!$(`#aperture${apertureCenter} .plot.frame`).length) {
        console.log('Create new plots');

        $(`#aperture${apertureCenter} .resizable.frame`).html(
          `<div class="plot frame" id="${framePlotId}"> </div>`);

          this.framePlots[i] = $.jqplot(
            framePlotId, [intensity, stddev], $.extend(plotOptions, {
              title: `Aperture ${i+1} - Last Frame - ${this.dataLabel}`,
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
        console.log('Refresh plots');

        this.framePlots[i].title.text = `Aperture ${i+1} - Last Frame - ${this.dataLabel}`;
        this.framePlots[i].series[0].data = intensity;
        this.framePlots[i].series[1].data = stddev;
        this.framePlots[i].resetAxesScale();

        // Refresh only on active tab
        if (i == activeTabIndex) {
          this.framePlots[i].replot();
        }

      }

    }
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

    /// Refresh tabs to update them
    $(`#${this.id}`).tabs('refresh');

    // Activate first tab if none is active
    if (this.activeTab == null) {
      this.activeTab = 0;
    }

    $(`#${this.id}`).tabs('option', 'active', this.activeTab);

  },

  /**
   * [description]
   * @param  {[type]} data [description]
   * @return {[type]}      [description]
   */
  updateStackArea: function(data) {

    this.stackSize = data.stack_size;
    console.log(`- Handling stack frame data - Stack size: ${this.stackSize}`);

    for (let i = 0; i < this.aperturesCenter.length; i++) {

      let apertureCenter = this.aperturesCenter[i];
      let stackPlotId = `stackPlot_${apertureCenter}`;
      let intensity = data.apertures[i].intensity;
      let stddev = data.apertures[i].stddev;
      let units = data.apertures[i].wavelength_units;
      let activeTabIndex = $(`#${this.id}`).tabs('option', 'active');

      $(`#aperture${apertureCenter} .info.stack`).html(
        getFrameInfo(data.filename, data.program_id)
      );

      // Check if plot containers exist
      if (!$(`#aperture${apertureCenter} .plot.stack`).length) {

        console.log('Create new plots');

        $(`#aperture${apertureCenter} .resizable.stack`).html(
          `<div class="plot stack" id="${stackPlotId}"> </div>`);

        this.stackPlots[i] = $.jqplot(
          stackPlotId, [intensity, stddev], $.extend(plotOptions, {
            title: `Aperture ${i + 1} - Stack Frame - Stack size: ${this.stackSize}`,
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

        console.log('Refresh plots');

        this.stackPlots[i].title.text = `Aperture ${i + 1} - Stack Frame - Stack size: ${this.stackSize}`;
        this.stackPlots[i].series[0].data = intensity;
        this.stackPlots[i].series[1].data = stddev;
        this.stackPlots[i].resetAxesScale();

        // Refresh only on active tab
        if (i == activeTabIndex) {
          this.stackPlots[i].replot();
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

function getApertureIndex(apertureCenter, listOfApertures) {

  let closestAperture = listOfApertures.reduce( function(prev, curr) {
    return (Math.abs(curr - apertureCenter) < Math.abs(prev - apertureCenter) ? curr : prev);
  });

  function getIndex(value) {
    return value === closestAperture;
  }

  return listOfApertures.findIndex(getIndex);

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


function notInApertureCenterList(aperture, pixelScale, listOfApertures) {

  let tolerance = 1.0;  // arcseconds
  let apertureInList = listOfApertures.some(
    function(l) {
      return (Math.abs(l - aperture) * pixelScale <= tolerance);
    }
  );

  i = getApertureIndex(aperture, listOfApertures);
  a = listOfApertures[i];

  if (apertureInList) {
    console.log(`- Updating plots for aperture: ${a} (${aperture})`);
  } else {
    console.log(`- New aperture: ${aperture}`, listOfApertures);
  }

  return !apertureInList;

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
