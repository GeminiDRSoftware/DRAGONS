
_gjs_cmdpipe = null;

 _gjs_cmdpipe = new GJSCommandPipe();
        


function GJSCommandPipe() {
    this.cmdq_url = "/cmdqueue.json"
    this.callbacks = {"stat":[]}
    
    this.iteratePump = function(data)
    {
        //console.log("iterate pump")
        // data should be a list of dictionaries, each dictionary is a message
        var gjs = this;
        $.each(data, function (ind) 
            {
                console.log("iterating message", JSON.stringify(this));
                msg = this;
                callbacks = gjs.callbacks[this["msgType"]];
                console.log("callbacks:"+JSON.stringify(callbacks));
                $.each(callbacks, function () {
                        this(msg);
                    });
            });
    }
    
    this.registerCallback = function(thetype, thefunction)
    {
        cblist = this.callbacks[thetype];
        cblist[cblist.length]= thefunction;
    }
}

function gempipe_cmdpump()
{
    $.ajax({url: "/cmdqueue.json",
            success: function (data) {
                    $("#cmd_display").html(JSON.stringify(data));
                    _gjs_cmdpipe.iteratePump(data);
                    setTimeout(gempipe_cmdpump, 1000);
                }
            }
            );
}

function register_stat_callback(stat)
    {
        _gjs_cmdpipe.registerCallback("stat", stat);
    }
