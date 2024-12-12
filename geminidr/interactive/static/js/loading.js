function openModal(message) {

  var modalOverlay = document.getElementsByClassName('modal_overlay');
  var modalTextBox = document.getElementsByClassName('modal_textbox');

  console.log(modalOverlay);
  console.log(modalTextBox);

  /* Set onclick event handler for all trigger elements */
  for(var i = 0; i < modalOverlay.length; i++) {
      modalOverlay[i].style.visibility = "visible";
      modalTextBox[i].innerHTML = `${message} <div class="loader"></div>`;
  }
}

function closeModal() {
  var modalOverlay = document.getElementsByClassName('modal_overlay');

  /* Set onclick event handler for all trigger elements */
  for(var i = 0; i < modalOverlay.length; i++) {
      modalOverlay[i].style.visibility = "hidden";
  }
}
