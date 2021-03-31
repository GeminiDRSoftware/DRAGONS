
/**
 * Sends key pressed events as ajax requests to Bokeh.
 */
function keyMapping(event) {
    // preventDefault causes the key not to pass to any widgets that are in
    // focus, such as a text input
    console.log("Pressed ", event.key)
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
        case 'p':
            $.ajax('/handle_key?key=p');
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
