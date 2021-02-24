function showHelp() {
    let helpModal = $( '.help-page' );
    console.log("Clicked on help button.");
    console.log(helpModal)
    helpModal.css("display", "block");
};

function closeHelp() {
    let helpModal = $( '.help-page' );
    console.log("Closing the help.");
    console.log(helpModal)
    helpModal.css("display", "none");
};