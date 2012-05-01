// Class for a database of metrics records returned by the adcc

// Constructor
function MetricsDatabase(key) {
    this.records = {};
}

// Add methods to prototype
MetricsDatabase.prototype = {
    constructor: MetricsDatabase,
    
    addRecord: function(key, record) {
	this.records[key] = record;
    },

    getRecord: function(key) {
	return this.records[key];
    },

    removeRecord: function(key) {
	delete this.records[key];
    },

    clearDatabase: function() {
	this.records = {};
    }
};