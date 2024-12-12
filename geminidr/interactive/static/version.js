
function setVersion(v) {
    console.log('setting version');
    var new_footer = 'Made with <a href="https://github.com/GeminiDRSoftware/DRAGONS/tree/v' + v + '">DRAGONS ' + v + '</a>&nbsp;&nbsp;';
    $('#footer').html(new_footer);
}
// Made with <a href="https://github.com/GeminiDRSoftware/DRAGONS/tree/release/2.2.0">DRAGONS v2.2.0</a>&nbsp;&nbsp;

window.onload = function(){
  $.ajax({
    url: "/version",
    context: document.body
  }).done(function(ver) {
    setVersion(ver);
  });
}
