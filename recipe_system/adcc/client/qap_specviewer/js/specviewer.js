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
  console.log(" Creating new object with id: ", id)

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
        console.log(data);
        console.log(data.apertures);

        addAperturesTabs(sViewer.id, data.apertures);

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
 * @param numberOfApertures
 */
function addAperturesTabs(parentId, numberOfApertures) {
  'use restrict';

  console.log(" Adding buttons to element with ID: ", parentId)

  // var tab = document.createElement("BUTTON");
  // var tabLabel = document.createTextNode("Aperture X");
  //
  // tab.setAttribute('class', 'tablinks');
  // tab.setAttribute('onclick', function() {console.log('Button clicked!')})
  // tab.appendChild(tabLabel)
  //
  // var parent = document.getElementById(parentId)
  // parent.appendChild(tab)

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
  var i, tabcontent, tablinks
  tabcontent = document.getElementsByClassName('tabcontent')
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = 'none'
  }
  tablinks = document.getElementsByClassName('tablinks')
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(' active', '')
  }
  document.getElementById(apertureId).style.display = 'block'
  evt.currentTarget.className += ' active'

  console.log(evt.currentTarget)
}
