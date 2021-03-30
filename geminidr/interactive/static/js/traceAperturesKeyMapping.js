
/**
 *
 */
function keyMapping(event) {
    // $("#plot_div").keydown(function(event) {
    // document.getElementById("plot_div").addEventListener("keydown", function(event){
    // doing whole document.  Or we could figure out how to target a bokeh-generated container div
    // preventDefault causes the key not to pass to any widgets that are in focus, such as a text input
    switch(event.key) {
        case 'a':
            $.ajax('/handle_key?key=a');
            event.preventDefault()
            break;
        case 'b':
            $.ajax('/handle_key?key=b');
            event.preventDefault()
            break;
        case 'r':
            $.ajax('/handle_key?key=r');
            event.preventDefault()
            break;
        case 'd':
            $.ajax('/handle_key?key=d');
            event.preventDefault()
            break;
        case 'e':
            $.ajax('/handle_key?key=e');
            event.preventDefault()
            break;
        case 'f':
            $.ajax('/handle_key?key=f');
            event.preventDefault()
            break;
        case '[':
            $.ajax('/handle_key?key=[');
            event.preventDefault()
            break;
        case ']':
            $.ajax('/handle_key?key=]');
            event.preventDefault()
            break;
        case 'l':
            $.ajax('/handle_key?key=l');
            event.preventDefault()
            break;
        case '*':
            $.ajax('/handle_key?key=*');
            event.preventDefault()
            break;
        case 'm':
            $.ajax('/handle_key?key=m');
            event.preventDefault()
            break;
        case 'u':
            $.ajax('/handle_key?key=u');
            event.preventDefault()
            break;
        //case 65:
        //  tabs.active = 0;
        //  event.preventDefault()
        //  break;
        //case 66:
        //  tabs.active = 1;
        //  event.preventDefault()
        //  break;
    }
}
