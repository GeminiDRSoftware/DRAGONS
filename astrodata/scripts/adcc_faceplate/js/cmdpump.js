function GJSCommandPipe(url) {
    if (url) {
	this.cmdq_url = url;
    } else {
	this.cmdq_url = "/cmdqueue.json";
    }
    this.callbacks = {};
    this.timeout = null;
    this.stop = true;
    this.timestamp = null;
    this.delay = 1000;
}
GJSCommandPipe.prototype = {
    constructor: GJSCommandPipe,

    iteratePump: function(data) {
	if (data.length==0) {
	    return;
	}

        // data should be a list of dictionaries, each dictionary is a message
        var gjs = this;
	var msg_types = {};
        $.each(data, function (ind) {
            var msg = this;
	    if (msg_types[msg["msgType"]]==undefined) {
		msg_types[msg["msgType"]] = [];
	    }
	    msg_types[msg["msgType"]].push(msg);
	}); // end each message

	$.each(msg_types, function(thetype) {
	    var msg_list = this;
	    var callbacks = gjs.callbacks[thetype];
	    $.each(callbacks, function () {
	        this(msg_list);
	    }); // end each callback
        }); // end each message type

	gjs.timestamp = data[data.length-1]['timestamp'];
    }, // end iteratePump
    
    registerCallback: function(thetype, thefunction) {
	if (this.callbacks[thetype]==undefined) {
	    this.callbacks[thetype] = [thefunction];
	} else {
	    this.callbacks[thetype].push(thefunction);
	}
    }, // end registerCallback

    pump: function() {
        var gjs = this;	
	$.ajax({type: "GET",
		data: {timestamp: gjs.timestamp},
		url: gjs.cmdq_url,
	        success: function (data) {
		    gjs.iteratePump(data);
		    if (!gjs.stop) {
			gjs.timeout = setTimeout(function(){
			    gjs.pump()
			}, gjs.delay); // end timeout
		    }
                }, // end success
		error: function() {
		    if (!gjs.stop) {
			gjs.timeout = setTimeout(function(){
			    gjs.pump()
			},gjs.delay); // end timeout
		    }
		} // end error
	}); // end ajax
    }, // end pump

    startPump: function(timestamp) {
	this.stop = false;
	this.timestamp = timestamp;
	this.pump();
    },

    stopPump: function() {
	this.stop = true;
	if (this.timeout) {
	    clearTimeout(this.timeout);
	    this.timeout = null;
	}
    } // end stopPump

}; // end prototype
