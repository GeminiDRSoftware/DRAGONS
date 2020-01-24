/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */
const specViewerJsonName = "/specqueue.json";


/**
 * Main component for SpecViewer.
 *
 * @param {JQuery Object} parentElement - element that will hold SpecViewer.
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

  /* Create empty page */
  this.parentElement.append(`<div id="${id}"><ul></ul></div>`);
  this.loadData();

  this.specPump = new SpectroPump();

} // end SpecViewer


SpecViewer.prototype = {

  constructor: SpecViewer,

  /**
   * Add navigation tabs based on how many apertures there is inside the
   * JSON file.
   *
   * @param parentId
   * @type parentId string
   *
   * @param numberOfApertures
   * @type number
   */
  addNavigationTab: function(parentId, numberOfApertures) {
    'use restrict';

    /* Add navigation tab container */
    var listOfTabs = $(`#${parentId} ul`);

    // Create buttons and add them to the navigation tab
    for (var i = 0; i < numberOfApertures; i++) {
      listOfTabs.append(`<li><a href="#aperture${i}">Aperture ${i}</a></li>`);
    }

  },

  /**
   * Add plots to the existing HTML elements.
   */
  addPlots: function(parentId, data) {
    'use restrict';

    var sViewer = this;

    var intensity = null;
    var stddev = null;

    var framePlots = [];
    var stackPlots = [];

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
      console.log("Selected tab ", tab.newTab.index())
      sViewer.resizeFramePlots(tab.newTab.index());
      sViewer.resizeStackPlots(tab.newTab.index());
    }

    // Call function to activate the tabs
    $( `#${parentId}` ).tabs('refresh');
    $( `#${parentId}` ).tabs('option', 'active', 0);    
    $( `#${parentId}` ).tabs( {'activate': selectTab} );

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
   * Resizes frame plots on different situations, like window resizing or 
   * when changing tabs.
   * 
   * @param activeTabIndex {number} 
   */
  resizeFramePlots: function (activeTabIndex) {

    console.log(`Resizing frame plot ${activeTabIndex}`);

    $(`framePlot${activeTabIndex}`).height(
      $(`framePlot${activeTabIndex}-resizable`).height() * 0.96
    );
    
    $(`framePlot${activeTabIndex}`).width(
      $(`framePlot${activeTabIndex}-resizable`).width() * 0.96
    );
    
    this.framePlots[activeTabIndex].replot({ resetAxes: true });

  },
  
  /**
   * Resizes frame plots on different situations, like window resizing or 
   * when changing tabs.
   * 
   * @param activeTabIndex {number} 
   */
  resizeStackPlots: function (activeTabIndex) {

    console.log(`Resizing stack plot ${activeTabIndex}`);

    $(`stackPlot${activeTabIndex}`).height(
      $(`stackPlot${activeTabIndex}-resizable`).height() * 0.96
    );
    
    $(`stackPlot${activeTabIndex}`).width(
      $(`stackPlot${activeTabIndex}-resizable`).width() * 0.96
    );
    
    this.stackPlots[activeTabIndex].replot({ resetAxes: true });

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
   */
  loadData: function() {
    'use restrict';

    // Reference to self to use in functions inside load
    var sViewer = this;

    $.ajax({
      type: "GET",
      url: specViewerJsonName,
      success: function(jsonData) {

        var data = JSON.parse(JSON.stringify(jsonData));

        // Call function to activate the tabs
        $(`#${sViewer.id}`).tabs();

        sViewer.addNavigationTab(sViewer.id, data.apertures.length);
        sViewer.addTabs(sViewer.id, data);
        sViewer.addPlots(sViewer.id, data);

        // Remove loading GIF
        $('.loading').remove();

      }, // end success
      error: function(e) {

        console.log('Could not receive json file');
        $( `#${sViewer.id}` ).html(e.responseText);

        // Remove loading GIF
        $('.loading').remove();

      } // end error
    }); // end ajax
  }, // end load

}; // end prototype


/**
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
