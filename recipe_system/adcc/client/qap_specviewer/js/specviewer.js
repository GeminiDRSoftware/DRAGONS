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
  this.element = parentElement;
  this.id = id;
  console.log("Creating new object with id: ", id)

  // Add a DIV containiner inside the parentElement with proper id
  var placeholder = document.createElement("DIV");
  placeholder.setAttribute('id', id);

  var specViewerParent = this.element.get(0);
  specViewerParent.appendChild(placeholder)

  // Placeholders for different elements
  this.apertureInfo = null;
  this.frameInfo = null;
  this.framePlot = null;
  this.stackInfo = null;
  this.stackPlot = null;

  // Placeholders for data information
  this.nApertures = 0;
  this.filename = null;
  this.programId = null;
  this.apertureDefinition = [];
  this.dispersion = [];
  this.frameWavelength = [];
  this.frameIntensity = [];
  this.frameVariance = [];

  // Create page
  this.load()

} // end SpecViewer


SpecViewer.prototype = {

  constructor: SpecViewer,

  load: function() {
    'use restrict';
    console.log('Calling "load" function')

    // Reference to self to use in functions inside load
    var sViewer = this;

    $.ajax({
      type: "GET",
      url: "/specframe.json",
      success: function(jsonData) {
        'use restrict';
        var data = JSON.parse(jsonData);
        console.log("JSON data received: ", data);
        console.log("Apertures received: ", data.apertures);
        console.log("Number of apertures received: ", data.apertures.length);

        addNavigationTab(sViewer.id, data.apertures.length);
        addTabs(sViewer.id, data);
        // addPlots(sViewer.id, data);

        // Add click event to show first aperture
        // $(".tablinks")[0].click()

        // Call function to activate the tabs
         $( `#${sViewer.id}` ).tabs( "refresh" );

      }, // end success
      error: function() {
        console.log('Could not receive json file');
      } // end error
    }); // end ajax
  }, // end load

}

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
function addNavigationTab(parentId, numberOfApertures) {
  'use restrict';
  console.log(`Adding ${numberOfApertures} buttons to element with ID: `, parentId);

  // Add navigation tab container
  var parent = $( `#${parentId}` );
  $( '<div class="tab" id="navigationTab"></div>' ).appendTo(parent);

  var navigationTab = $('#navigationTab');
  navigationTab.tabs();

  // Create empty list of tabs
  $( '<ul></ul>' ).appendTo(navigationTab);
  var listOfTabs = $('#navigationTab ul');

  // Create buttons and add them to the navigation tab
  for (var i = 0; i < numberOfApertures; i++) {
    console.log(`Create button for Aperture${i + 1}`);
    listOfTabs.append(`<li><a href="#aperture${i + 1}">Aperture ${i + 1}</a></li>`)
  }

}


/**
 * Add tabs containing plots and information on each aperture.
 *
 * @param parentId
 * @type parentId string
 *
 * @param data
 * @type data object
 */
function addTabs(parentId, data){
  'use restrict';
  var parent = $( `#${parentId}` );

  for (var i = 0; i < data.apertures.length; i++) {

    var aperture = data.apertures[i];
    console.log(`Aperture ${i + 1} definition`, aperture);

    parent.append(
      `<div id="aperture${i + 1}" class="tabcontent"> </div>`
    );

    $(`#aperture${i + 1}`).append(
      '<div class="apertureInfo">' +
      '  <span>' +
      `    <b>Aperture definition:</b> ${aperture.center} (${aperture.lower}, ${aperture.upper})` +
      '  </span>' +
      '</div>'
    );

    // var dispersion = document.createElement("SPAN");
    // dispersion.style.paddingLeft = "68px"
    // dispersion.innerHTML =
    //   `<b>Dispersion:</b> ${aperture.dispersion} nm/px`
    // apertureInfo.appendChild(dispersion);
    //
    // var lastFrameInfo = document.createElement("DIV");
    // lastFrameInfo.className = "frameInfo";
    // lastFrameInfo.innerHTML =
    //   `Last frame - ${data.filename} - ${data.programId}`;
    // tab.appendChild(lastFrameInfo);
    //
    // var lastFramePlot = document.createElement("DIV");
    // lastFramePlot.className = "framePlot";
    // tab.appendChild(lastFramePlot);
    //
    // var stackFrameInfo = document.createElement("DIV");
    // stackFrameInfo.className = "stackInfo";
    // stackFrameInfo.innerHTML =
    //   `Stack frame - ${data.filename} - ${data.programId}`;
    // tab.appendChild(stackFrameInfo);
    //
    // var stackFramePlot = document.createElement("DIV");
    // stackFramePlot.className = "stackPlot";
    // tab.appendChild(stackFramePlot);

  }
}

/**
 * Add plots to the existing HTML elements.
 *
 */
function addPlots(parentId, data) {
  'use restrict';

  var tabcontent = document.getElementsByClassName('tabcontent');

  for (var i=0; i < data.apertures.length; i++){
    console.log("Add plot for aperture ", i);

    var wavelength = data.apertures[i].wavelength;
    var intensity = data.apertures[i].intensity;
    var variance = data.apertures[i].variance;

    var framePlotArea = tabcontent[i].getElementsByClassName('framePlot')[0];
    framePlotArea.id = `framePlotArea${i+1}`;
    console.log(framePlotArea);

    var stackPlotArea = tabcontent[i].getElementsByClassName('stackPlot')[0];
    stackPlotArea.id = `stackPlotArea${i+1}`;

    var cosPoints = [];
    for (var i=0; i<2*Math.PI; i+=0.1){
      cosPoints.push([i, Math.cos(i)]);
    }

    var framePlot = $.jqplot (framePlotArea.id, [cosPoints]);
    // var stackPlot = $.jqplot (stackPlotArea.id, [[3,7,9,1,5,3,8,2,5]]);

  }

}


/**
 * This function handles the onclick event for the tab buttons at the top of
 * the main page. It starts hiding every `tabcontent` element, then cleaning
 * up the `active` class name from every `tablinks` element and, finally,
 * adds back the `active` class name to the events owner.
 *
 * @parameter evt
 * @parameter apertureId
 */
function openAperture(evt, apertureId) {

  console.log(`Pressed button related to ${apertureId}`);

  var tabcontent = document.getElementsByClassName('tabcontent');
  for (var i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = 'none';
  }

  var tablinks = document.getElementsByClassName('tablinks')
  for (var i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(' active', '')
  }

  document.getElementById(apertureId).style.display = 'block'
  evt.currentTarget.className += ' active'
}
