
var help_popup = null;

/**
 * Displays the detailed help in a new window
 * @param help_title - Title for the Help window
 */
function openHelpPopup(help_title) {

    if (help_popup == null || help_popup.closed) {
        help_popup = window.open(
            window.location + "help",
            help_title,
            "resizable=yes,scrollbars=yes,status=yes,height=800,width=800");
    } else {
        help_popup.focus();
    };
};

/**
 * Closes the Help Window when the Main Window closes
 */
window.onbeforeunload = function() {
    help_popup.close();
};
