/* start module: os */
var os;
$pyjs.loaded_modules['os'] = function (__mod_name__) {
	if($pyjs.loaded_modules['os'].__was_initialized__) return $pyjs.loaded_modules['os'];
	os = $pyjs.loaded_modules["os"];
	os.__was_initialized__ = true;
	if ((__mod_name__ === null) || (typeof __mod_name__ == 'undefined')) __mod_name__ = 'os';
	var __name__ = os.__name__ = __mod_name__;
	os.__track_lines__ = new Array();
	os.__track_lines__[1] = "os.py, line 1:\n    ";
	os.__track_lines__[18] = "os.py, line 18:\n    return bs";
	os.__track_lines__[3] = "os.py, line 3:\n    def urandom(n):";
	os.__track_lines__[17] = "os.py, line 17:\n    raise NotImplementedError(\"/dev/urandom (or equivalent) not found\")";


	$pyjs.track.module='os';
	$pyjs.track.lineno=1;
	$pyjs.track.lineno=3;
	os['urandom'] = function(n) {

		$pyjs.track={module:'os',lineno:3};$pyjs.trackstack.push($pyjs.track);
		$pyjs.track.module='os';
		$pyjs.track.lineno=3;
		$pyjs.track.lineno=17;
		throw ((function(){var $pyjs_dbg_1_retry = 0;
try{var $pyjs_dbg_1_res=pyjslib['NotImplementedError'](String('/dev/urandom (or equivalent) not found'));}catch($pyjs_dbg_1_err){
    if ($pyjs_dbg_1_err.__name__ != 'StopIteration') {
        var save_stack = $pyjs.__last_exception_stack__;
        sys.save_exception_stack();
        var $pyjs_msg = "";

        try {
            $pyjs_msg = "\n" + sys.trackstackstr();
        } catch (s) {};
        $pyjs.__last_exception_stack__ = save_stack;
        if ($pyjs_msg !== $pyjs.debug_msg) {
            pyjslib['debugReport']("Module os at line 17 :\n" + $pyjs_dbg_1_err + $pyjs_msg);
            $pyjs.debug_msg = $pyjs_msg;
            debugger;
        }
    }
    switch ($pyjs_dbg_1_retry) {
        case 1:
            $pyjs_dbg_1_res=pyjslib['NotImplementedError'](String('/dev/urandom (or equivalent) not found'));
            break;
        case 2:
            break;
        default:
            throw $pyjs_dbg_1_err;
    }
}return $pyjs_dbg_1_res})());
		$pyjs.track.lineno=18;
		$pyjs.track.lineno=18;
		var $pyjs__ret = os.bs;
		$pyjs.trackstack.pop();$pyjs.track=$pyjs.trackstack.pop();$pyjs.trackstack.push($pyjs.track);
		return $pyjs__ret;
	};
	os['urandom'].__name__ = 'urandom';

	os['urandom'].__bind_type__ = 0;
	os['urandom'].__args__ = [null,null,['n']];
	return this;
}; /* end os */


/* end module: os */


