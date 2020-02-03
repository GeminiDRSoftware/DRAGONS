/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */
const specViewerJsonName = "/specqueue.json";


/**
 * Main component for SpecViewer.
 *
 * @param {object} parentElement - element that will hold SpecViewer.
 * @param {string} id - name of the ID of the SpecViewer div container.
 */
function SpecViewer(parentElement, id) {
  'use strict';

  // Creating empty object
  this.parentElement = parentElement;
  this.id = id;

  // Placeholders for different elements
  this.activeTab = null;
  this.framePlots = [];
  this.stackPlots = [];

  this.aperturesCenter = [];
  this.dataLabel = null;
  this.groupId = null;

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

  // Add event handler to tabs
  $(`#${id}`).tabs({activate: this.onTabChange});

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
   * @deprecated
   *
   * Add navigation tabs based on how many apertures there is inside the
   * JSON file.
   *
   * @param {string} parentId
   * @param {number} numberOfApertures
   */
  addNavigationTab: function(parentId, numberOfApertures) {
    'use restrict';

    /* Add navigation tab container */
    let listOfTabs = $(`#${parentId} ul`);

    // Create buttons and add them to the navigation tab
    for (let i = 0; i < numberOfApertures; i++) {
      listOfTabs.append(`<li><a href="#aperture${i}">Aperture ${i}</a></li>`);
    }

  },

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

    // Resize plot area on window resize
    $(window).resize(function onWindowResize() {
      var activeTabIndex = $(`#${sViewer.id}`).tabs('option', 'active');
      sViewer.resizeFramePlots(activeTabIndex);
      sViewer.resizeStackPlots(activeTabIndex);

    });

    // Add button for reset zoom
    framePlots.map(function(p, i) {
      $(`#resetZoomFramePlot${i}`).click(function() {
        console.log(`Reset zoom of frame plot #${i}.`);
        p.resetZoom();
      });
    });

    stackPlots.map(function(p, i) {
      $(`#resetZoomStackPlot${i}`).click(function() {
        console.log(`Reset zoom of stack plot #${i}.`);
        p.resetZoom();
      });
    });

  },

  /**
   * Add/refresh tab content for incomming data.
   * @type {object} data Incomming JSON data
   */
  newTabContent: function(data) {

    for (let i = 0; i < data.apertures.length; i++) {

      let aperture = data.apertures[i];

      // Add empty tab container if it does not exist
      // !!! ToDo: improve matching criterium !!!
      if (!this.aperturesCenter.includes(aperture.center)) {

        $(`#${this.id}`).append(
          `<div id="aperture${aperture.center}" class="tabcontent">
            <div class="apertureInfo"> </div>
            <div class="frameInfo"> </div>
            <div id="framePlot${aperture.center}-resizable" class="ui-widget-content resizable"> </div>
            <div class="stackInfo"> </div>
            <div id="stackPlot${aperture.center}-resizable" class="ui-widget-content resizable"> </div>
          </div>`
        );

        this.aperturesCenter.push(aperture.center);

      }

      // Add content to the container
      $(`#aperture${aperture.center} .apertureInfo`).html(
        getApertureInfo(aperture)
      );

      $(`#aperture${aperture.center} .frameInfo`).html(
        getFrameInfo("", "")
      );

      $(`#aperture${aperture.center} .stackInfo`).html(
        getStackInfo("", "")
      );

    }

  },

  /**
   * Add tabs containing plots and information on each aperture.
   *
   * @param parentId
   * @type parentId string
   *
   * @param data
   * @type data object
   */
  addTabs: function(parentId, data) {

    'use restrict';
    var parent = $(`#${parentId}`);

    for (var i = 0; i < data.apertures.length; i++) {

      var aperture = data.apertures[i];

      const apertureTabContent = `
        <div id="aperture${i}" class="tabcontent">

          <div class="apertureInfo">
            <span>
              <b>Aperture definition:</b>
                <span class="app-info-field" title="Aperture center"> ${aperture.center} px </span> (
                <span class="app-info-field" title="Lower aperture limit"> ${aperture.lower} px </span>,
                <span class="app-info-field" title="Upper aperture limit"> ${aperture.upper} px </span>)
            </span>
            <span style="padding-left: 10%">
              <b>Dispersion:</b> ${aperture.dispersion} nm/px
            </span>
          </div>

          <div class="frameInfo">
            <div class="d-table w-100">
              <p class="d-table-cell">
                Latest frame - ${data.filename} - ${data.programId}
              </p>
              <div class="d-table-cell tar">
                <button class="ui-button ui-widget ui-corner-all" id="resetZoomFramePlot${i}" title="Reset zoom">
                  <img class="zoom-reset" src="/qlook/images/zoom_reset_48px.png"></img>
                </button>
              </div>
            </div>
          </div>

          <div id="framePlot${i}-resizable" class="ui-widget-content resizable">
            <div class="framePlot" id="framePlot${i}">
            </div>
          </div>

          <div class="stackInfo">
            <div class="d-table w-100">
              <p class="d-table-cell">
                Stack frame - ${data.filename} - ${data.programId}
              </p>
              <div class="d-table-cell tar">
                <button id="resetZoomStackPlot${i}" class="ui-button ui-widget ui-corner-all" title="Reset zoom">
                    <img class="zoom-reset" src="/qlook/images/zoom_reset_48px.png"></img>
                </button>
              </div>
            </div>
          </div>

          <div id="stackPlot${i}-resizable" class="ui-widget-content resizable">
            <div class="stackPlot" id="stackPlot${i}">
            </div>
          </div>

        </div>
      `;

      parent.append(apertureTabContent);
    } // end for

  }, // end addTabs

  /**
   * Query server for JSON file and start to populate page.
   * This function is the registered callback on the command pump.
   *
   * @param jsonData {object}
   */
  loadData: function(jsonData) {
    'use restrict';

    let now = Date(Date.now());

    console.log(`\nReceived new JSON data list on\n ${now.toString()}`);

    // Remove loading
    $('.loading').remove();

    for (let i = 0; i < jsonData.length; i++) {

      if (jsonData[i].data_label === this.dataLabel) {
        console.log(`- OLD data label: ${jsonData[i].data_label}`);
      } else {

        let jsonElement = jsonData[i];

        console.log(`- NEW data label: ${jsonElement.data_label}`);
        this.dataLabel = jsonElement.data_label;

        if (jsonElement.group_id === this.groupId) {
          console.log(`- MATCHING group id: ${jsonElement.group_id}`);
        } else {

          console.log(`- NEW group id: ${jsonElement.group_id}`);
          this.groupId = jsonElement.group_id;

          // Clear navigation tabs and aperture list
          $(`#${this.id} ul`).empty();
          this.aperturesCenter = [];

          // Clear tab contents
          $('.tabcontent').remove();

          // Add tab content
          this.newTabContent(jsonElement);
          this.updateNavigationTab(jsonElement);

          console.log(jsonElement.is_stack);

          if (jsonElement.is_stack) {
            this.updateStackArea(jsonElement);
          } else {
            this.updateFrameArea(jsonElement);
          }

        }

      }
    }

    //  sViewer.addNavigationTab(sViewer.id, data.apertures.length);
    //	sViewer.addTabs(sViewer.id, data);
    //	sViewer.addPlots(sViewer.id, data);

    // Remove loading
    $('.loading').remove();

  }, // end load

  /**
   * Handle event onTabChange; aka as $.tabs({activate: ...}).
   */
  onTabChange: function(event, tab) {

    var newIndex = tab.newTab.index();

    // sViewer.resizeFramePlots(tab.newTab.index());
    // sViewer.resizeStackPlots(tab.newTab.index());
  },

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

  }, // end start

  /**
   * [updateNavigationTab description]
   * @type {[type]}
   */
  updateFrameArea: function(data) {

    console.log('- Handling latest frame data');
    for (let i = 0; i < this.aperturesCenter.length; i++) {
      let apertureCenter = this.aperturesCenter[i];

      $(`#aperture${apertureCenter} .frameInfo`).html(
        getFrameInfo(data.filename, data.program_id)
      );

    }

  },

  /**
   * Updates the navigation tab based on matching aperture center.
   *
   * @param jsonElement {object}
   */
  updateNavigationTab: function(jsonElement) {

    // Add navigation tab container
    let navTabContainer = $(`#${this.id} ul`);

    this.aperturesCenter.sort();

    // Create buttons and add them to the navigation tab
    for (let i = 0; i < this.aperturesCenter.length; i++) {
      navTabContainer.append(`<li><a href="#aperture${this.aperturesCenter[i]}">Aperture ${i + 1}</a></li>`);
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
   * [updateNavigationTab description]
   * @type {[type]}
   */
  updateStackArea: function(data) {
    console.log('Handling stack frame data');
  },

}; // end prototype


/**
 * @deprecated Now that input JSON file comes in the correct format.
 *
 * Read two arrays and convert then into a single [x, y] array to be used in
 * plots.
 *
 * @param  {array} x One dimensional array with the X coordinates.
 * @param  {array} y One dimensional array with the Y coordinates.
 *
 * @return {array} One dimensional arrays containing [x, y] points
 */
function buildSeries(x, y) {
  var temp = [];
  for (var i = 0; i < x.length; i++) {
    temp.push([x[i], y[i]]);
  }
  return temp;
}

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
 * Returns the last frame info div element using a template.
 *
 * @param filename {string} name of the filename.
 * @param programId {string} program Id code.
 */
function getFrameInfo(filename, programId) {

  if (filename === '') {
    filename = '<span style="color: #999"> (Not available yet) </span>';
  }

  if (programId === '') {
    programId = '<span style="color: #999"> (Not available yet) </span>';
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
    filename = '<span style="color: #999"> (Not available yet) </span>';
  }

  if (programId === '') {
    programId = '<span style="color: #999"> (Not available yet) </span>';
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
 * Options to be used by the plots
 */
plotOptions = {

  axesDefaults: {
    alignTicks: true,
  },

  axes: {
    xaxis: {
      label: "Wavelength [\u212B]", // escaped Angstrom symbol
      labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
    },
    yaxis: {
      label: "Intensity [e\u207B]", // escaped superscript minus
      labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
    },
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
      color: '#ff7f0e',
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
