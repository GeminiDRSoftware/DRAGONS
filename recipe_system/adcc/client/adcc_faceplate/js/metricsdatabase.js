// Class for a database of metrics records returned by the adcc
// (can actually be used for other types of records too; should probably
// change the name)

// Constructor
function MetricsDatabase(key) {
    this.records = {};
}

// Add methods to prototype
MetricsDatabase.prototype = {
    constructor: MetricsDatabase,

    addRecord: function(key, record) {
	if (this.records[key]) {
	    for (var i in record) {
		this.records[key][i] = record[i];
	    }
	} else {
	    this.records[key] = record;
	}
    },

    getRecord: function(key) {
	return this.records[key];
    },

    getRecordList: function(key) {
	var record_list = [];
	for (var i in this.records) {
	    record_list.push(this.records[i]);
	}
	return record_list;
    },

    removeRecord: function(key) {
	delete this.records[key];
    },

    clearDatabase: function() {
	this.records = {};
    },

    size: function() {
	count = 0;
	for (var i in this.records) {
	    count++;
	}
	return count;
    }
};
