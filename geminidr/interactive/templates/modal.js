function openModal() {
  var modalTrigger = document.getElementsByClassName('modal_overlay');

  /* Set onclick event handler for all trigger elements */
  for(var i = 0; i < modalTrigger.length; i++) {
      modalTrigger[i].style.visibility = "visible";
  }
}

function closeModal() {
  var modalTrigger = document.getElementsByClassName('modal_overlay');

  /* Set onclick event handler for all trigger elements */
  for(var i = 0; i < modalTrigger.length; i++) {
      modalTrigger[i].style.visibility = "hidden";
  }
}
