/**
 * QAP SpecViewer Command Pump
 */
class SpectroPump {

  /**
   * [constructor description]
   * @param {[type]} url [description]
   */
  constructor(url) {

    console.log("Created SpectroPump object")
    this.url = url || "/cmdqueue.json";
    this.callbacks = {};
    this.timeout = null;
    this.stop = true;
    this.timestamp = null;
    this.msgtype = null;
    this.delay = 1000;

  };

  /**
   * [iteratePump description]
   * @param  {[type]} data [description]
   * @return {[type]}      [description]
   */
  iteratePump(data) {

      if (!data || data.length == 0) {
        return;
      }

      // data should be a list of dictionaries, each dictionary is a message
      var sPump = this;
      var msg_types = {};

      $.each(data, function(ind) {
        var msg = this;
        if (msg_types[msg["msgtype"]] == undefined) {
          msg_types[msg["msgtype"]] = [];
        }
        msg_types[msg["msgtype"]].push(msg);
      }); // end each message

      $.each(msg_types, function(thetype) {
        var msg_list = this;
        var callbacks = gjs.callbacks[thetype];
        if (callbacks) {
          $.each(callbacks, function() {
            this(msg_list);
          }); // end each callback
        }
      }); // end each message type

      gjs.timestamp = data[data.length - 1]['timestamp'];
    };

    /**
     * [registerCallback description]
     * @param  {string} thetype      [description]
     * @param  {function} thefunction [description]
     */
    registerCallback(thetype, thefunction) {
      if (this.callbacks[thetype] == undefined) {
        this.callbacks[thetype] = [thefunction];
      } else {
        this.callbacks[thetype].push(thefunction);
      };
    };

    /**
     * [pump description]
     * @return {[type]} [description]
     */
    pump() {

      var gjs = this;

      if (this.timeout) {
        clearTimeout(this.timeout);
        this.timeout = null;
      }

      $.ajax({
        type: "GET",
        data: { timestamp: gjs.timestamp, msgtype: gjs.msgtype},
        url: gjs.cmdq_url,
        success: function(data) {
          gjs.iteratePump(data);
          if (!gjs.stop) {
            gjs.timeout = setTimeout(function() {
              gjs.pump()
            }, gjs.delay); // end timeout
          }
        }, // end success
        error: function() {
          if (!gjs.stop) {
            gjs.timeout = setTimeout(function() {
              gjs.pump()
            }, gjs.delay); // end timeout
          }
        } // end error
      }); // end ajax
    };

    /**
     * Starts the loop that will query the ADCC server periodically.
     * @param  {float} timestamp [description]
     * @param  {string} msgtype   [description]
     */
    startPump(timestamp, msgtype) {
      this.stop = false;
      this.timestamp = timestamp;
      this.msgtype = msgtype;
      this.pump();
    };

    /**
     * Stops the loop that will query the ADCC server periodically.
     */
    stopPump() {
      this.stop = true;
      if (this.timeout) {
        clearTimeout(this.timeout);
        this.timeout = null;
      }
    };

}
