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
        addPlots(sViewer.id, data);

        // Add click event to show first aperture
        $(".tablinks")[0].click()

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

  // Create navigation tab container
  var navigationTab = document.createElement("DIV");
  navigationTab.className = "tab";
  navigationTab.id = "navigationTab";

  // Add navigation tab to the parent element
  var parent = document.getElementById(parentId);
  parent.appendChild(navigationTab);

  // Create buttons and add them to the navigation tab
  for (var i = 0; i < 3; i++) {
    console.log(`Create button for Aperture${i + 1}`);
    var tabLink = document.createElement("BUTTON");
    tabLink.className = "tablinks";
    tabLink.innerHTML = `Aperture ${i + 1}`;

    // Workaround using Immediately-Invoked Function Expressions
    tabLink.onclick = (function(apertureIndex){
      return function() {
        apertureId = `Aperture${apertureIndex}`;
        openAperture(event, apertureId);
      }
    })(i + 1);

    navigationTab.appendChild(tabLink);
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
  var parent = document.getElementById(parentId);

  for (var i = 0; i < data.apertures.length; i++) {

    var aperture = data.apertures[i];
    console.log(`Aperture ${i + 1} definition`, aperture);

    var tab = document.createElement("DIV");
    tab.id = `Aperture${i + 1}`;
    tab.className = "tabcontent";
    parent.appendChild(tab);

    var apertureInfo = document.createElement("DIV");
    apertureInfo.className = "apertureInfo";
    tab.appendChild(apertureInfo)

    var apertureDefinition = document.createElement("SPAN");
    apertureDefinition.innerHTML =
      `<b>Aperture definition:</b> ${aperture.center} px ` +
      `(${aperture.lower}, ${aperture.upper})`;
    apertureInfo.appendChild(apertureDefinition);

    var dispersion = document.createElement("SPAN");
    dispersion.style.paddingLeft = "68px"
    dispersion.innerHTML =
      `<b>Dispersion:</b> ${aperture.dispersion} nm/px`
    apertureInfo.appendChild(dispersion);

    var lastFrameInfo = document.createElement("DIV");
    lastFrameInfo.className = "frameInfo";
    lastFrameInfo.innerHTML =
      `Last frame - ${data.filename} - ${data.programId}`;
    tab.appendChild(lastFrameInfo);

    var lastFramePlot = document.createElement("DIV");
    lastFramePlot.className = "framePlot";
    tab.appendChild(lastFramePlot);

    var stackFrameInfo = document.createElement("DIV");
    stackFrameInfo.className = "stackInfo";
    stackFrameInfo.innerHTML =
      `Stack frame - ${data.filename} - ${data.programId}`;
    tab.appendChild(stackFrameInfo);

    var stackFramePlot = document.createElement("DIV");
    stackFramePlot.className = "stackPlot";
    tab.appendChild(stackFramePlot);

  }
}

/**
 * Add plots to the existing HTML elements.
 *
 */
function addPlots(parentId, data) {
  'use restrict';

  var parent = document.getElementById(parentId);

  for (var i=0; i < data.apertures.length; i++){
    console.log("Add plot for aperture ", i);

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
