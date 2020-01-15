/**
 * DRAGONS
 * Quality Assessment Pipeline - Spectrum Viewer
 *
 */

/**
 * Main component for SpecViewer.
 *
 * @param element
 * @param id
 */
function SpecViewer(parentElement, id) {
  'use strict';

  // Creating empty object
  this.parent = parentElement;
  this.id = id;

  // Placeholders for different elements
  this.framePlots = [];

  /* Create empty page */
  parentElement.append(`<div id="${id}"><ul></ul></div>`);
  this.loadData();

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

    /* Create buttons and add them to the navigation tab */
    for (var i = 0; i < numberOfApertures; i++) {
      listOfTabs.append(`<li><a href="#aperture${i}">Aperture ${i}</a></li>`);
    };
  },

  /**
   * Add plots to the existing HTML elements.
   */
  addPlots: function(parentId, data) {
    'use restrict';

    var plots = [];
    var stackPlots = []

    for (var i = 0; i < data.apertures.length; i++) {

      // Adding plot for frame
      var intensity = buildSeries(
        data.apertures[i].wavelength, data.apertures[i].intensity);

      var variance = buildSeries(
        data.apertures[i].wavelength, data.apertures[i].variance);

      plots[i] = $.jqplot(
        `framePlot-${i}`, [intensity, variance], $.extend(plotOptions, {
          title: `Aperture ${i} - Last Frame`,
        }));

      // Adding plots for stack
      var intensity = buildSeries(
        data.stackApertures[i].wavelength, data.stackApertures[i].intensity);

      var variance = buildSeries(
        data.stackApertures[i].wavelength, data.stackApertures[i].variance);

      stackPlots[i] = $.jqplot(
        `stackPlot-${i}`, [intensity, variance], $.extend(plotOptions, {
          title: `Aperture ${i} - Stack Frame`,
        }));

    }

    // Resize plots on window resize
    $(window).resize(function() {
      plots.map(function(p) {
        p.replot({
          resetAxes: true
        });
      });
    });

    $(window).resize(function() {
      stackPlots.map(function(p) {
        p.replot({
          resetAxes: true
        });
      });
    });

    // Display plots on tab change
    $(`#${parentId}`).bind('tabsshow', function(event, ui) {
      plots[ui.index].replot();
    });

    $(`#${parentId}`).bind('tabsshow', function(event, ui) {
      stackPlots[ui.index].replot();
    });

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
              ${aperture.center} (${aperture.lower}, ${aperture.upper})
            </span>
            <span style="padding-left: 10%">
              <b>Dispersion:</b> ${aperture.dispersion} nm/px
            </span>
          </div>

          <div class="frameInfo">
            Latest frame - ${data.filename} - ${data.programId}
          </div>

          <div class="framePlot" id="framePlot-${i}">
          </div>

          <div class="stackInfo">
            Stack frame - ${data.filename} - ${data.programId}
          </div>

          <div class="stackPlot" id="stackPlot-${i}">
          </div>

        </div>
      `;

      parent.append(apertureTabContent)
    }; // end for

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
      url: "/specframe.json",
      success: function(jsonData) {
        'use restrict';
        var data = JSON.parse(jsonData);

        // Call function to activate the tabs
        $(`#${sViewer.id}`).tabs();

        sViewer.addNavigationTab(sViewer.id, data.apertures.length);
        sViewer.addTabs(sViewer.id, data);
        sViewer.addPlots(sViewer.id, data);

        // Call function to activate the tabs
        $(`#${sViewer.id}`).tabs("refresh");
        $(`#${sViewer.id}`).tabs({
          active: 0
        });

        /* Remove loading GIF */
        $( '.loading' ).remove();

      }, // end success
      error: function() {
        console.log('Could not receive json file');
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
  var temp = []
  for (var i = 0; i < x.length; i++) {
    temp.push([x[i], y[i]]);
  }
  return temp
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
      pad: 1.01,
    },
    yaxis: {
      label: "Flux [???]",
    },
  },

  seriesDefaults: {
    lineWidth: 1,
    markerOptions: {
      size: 1
    },
    renderer: $.jqplot.LineRenderer,
  },

  series: [{
      color: '#1f77b4',
      label: 'Intensity',
    },
    {
      color: '#ff7f0e',
      label: 'Variance'
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
    show: true,
    zoom: true,
    constrainOutsideZoom: false,
    looseZoom: true,
    showTooltip: false,
    useAxesFormatters: false
  },

}
