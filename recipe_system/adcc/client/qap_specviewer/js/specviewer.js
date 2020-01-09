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
function SpecViewer(element, id) {
  'use strict';

  this.element = element;
  this.id = id;

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

    $.ajax({
      type: "GET",
      url: "/specframe.json",
      success: function(jsonData) {
        'use restrict';
        var data = JSON.parse(jsonData);
        console.log(data);
        console.log(data.apertures);
      }, // end success
      error: function() {
        console.log('Could not receive json file');
      } // end error
    }); // end ajax
  }, // end load

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
