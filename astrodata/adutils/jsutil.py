class JSDiv:
    def div(self):
        return "<div>override JSDiv.div()</div>"
        
class JSAccord(JSDiv):
    def div(self):
        return """
            <div class="gem_accord" style="display:none">
            <script type="text/javascript">
            function dict2accordian(accord, appTo, startOpen )
            {
                if (typeof(startOpen) == "undefined")
                {
                    var block = "block";
                    var subblock = "none"; 
                    var substartopen = false;
                }
                else
                {
                    if (startOpen)
                    {
                        var block = "block";
                        var subblock = block; 
                        var substartopen = startOpen
                    }                        
                    else
                    {
                        var block        = "none";
                        var subblock     = block; 
                        var substartopen = startOpen;

                    }
                }
                if (typeof(appTo) == "undefined")
                    appTo = null;
                // turns nested dict into collapsable hierarchical set of divs
                // distinguishes between typeof == object for sub-directories
                // and typeof == string which is an element
                            /*    
                            { category: name
                              subcats: [ {} {} {} ]
                              members: [ {} {} {} ]
                            }
                            */  
                var ndiv_html = '<div style="'
                                    + 'display:'+block+';'
                                    + 'border:solid black 1px; padding:2px;margin:1px;margin-left:1em"' 
                                    + 'class="recipies_'
                                    + accord.cat_name
                                    + '">'
                                    + '</div>'
                var title_ndiv_html = '<div style="margin-left:1em"><a href="javascript:void(0)" onclick="'
                                + "$('.recipies_"
                                    +accord.cat_name
                                    +"').toggle()"
                                + '">'
                                + "Recipe Folder: "
                                + accord.cat_name
                                + '</a></div>'
                
                var title = $(title_ndiv_html);
                if (appTo)
                {
                    appTo.append(title);
                }                  
                //console.log("js25:"+ndiv_html);
                var ndiv = $(ndiv_html);
                //for (key in accord)
                //{
                //    console.log("js27:"+ key);
                //}
                if ("members" in accord)
                {
                    accord.members.sort( function (a,b) {
                        var aname  = a.name.toLowerCase();
                        var bname  = b.name.toLowerCase();
                        if (aname > bname) { return 1;};
                        if (aname < bname) { return -1;};                        
                        return 0;                        
                        });
                    for (var i = 0; i < accord.members.length; i++)
                    {
                        var mem = accord.members[i];
                        // console.log("js26:"+JSON.stringify(mem));
                        var mem_span ='<span title="'
                                            + mem.path
                                            + '"'
                                            + ' class="recipe_'
                                            + mem.name
                                            + '">'
                                            + mem.name 
                                            + ' (<a href="javascript: void(0)" onclick="editFile('
                                            + "'"
                                            + mem.path
                                            + "')"+ '">'
                                            + "edit) </a>"
                                            + '</span></br>'; 
                        //console.log("js69:"+mem_span);
                        ndiv.append($(mem_span));
                    }
                }
                if ("subcats" in accord)
                {
                    for (var i = 0; i < accord.subcats.length; i++)
                    {
                        var subcat = accord.subcats[i];
                        var nsdiv = dict2accordian(subcat, ndiv, substartopen);
                        ndiv.append(nsdiv);
                    }
                }
                //console.log("js28:"+ndiv.html());
                return ndiv;
            }
            </script>
            </div>
            """

class JSTypes(JSDiv):
    def div(self):
        return """<div class="lp_types" style="width:20%;float:left">
                   <script type="text/javascript">
                   function populateTypesDiv(typemap)
                   {
                        // console.log("ju10:"+JSON.stringify(typemap))
                        var typl = typemap["types"];
                        buff = "<b>Types within <tt>"
                             + typemap["package"]
                             + "</tt> Package</b><br/>";
                        for (var i =0 ; i < typl.length; i++)
                        {
                            buff += typl[i]; 
                            buff += ' (<a href="javascript: void(0)" onclick="ajaxLink(';
                            buff += "'"
                            buff += typemap["type_meta"][typl[i]].edit_link;
                            buff += "')"+ '">';
                            buff += "edit"
                            buff += "</a>)<br/>";
                        }
                        //console.log("16:"+buff);
                        $($(".lp_types_content")[0]).html(buff);
                   }
                   </script>
                   <div class = "lp_types_content" style="border:solid black 1px;padding:2px;margin:3px">
                   </div>                               
                </div> 
                """
class JSDescriptors(JSDiv):
    def div(self):
         return """<div class="lp_descriptors" style="float:left;width:38%">
                   <script type="text/javascript">
                   function populateDescriptorsDiv(desmap)
                   {
                        //alert(JSON.stringify(desmap));
                        var descmap = desmap["descriptors"]
                        buff = "<b>Descriptor Calculators within <tt>"
                                +desmap["package"]
                                +"</tt> Package</b><br/>";
                        buff += "<table>";
                        buff += "<tr><th>Type</th><th>Calculator Class</th></tr>"
                        for (var i = 0; i < descmap.length; i++)
                        {
                            var desc = descmap[i];
                            
                            line =      "<tr>"
                                        +"<td>" + desc.type + "</td>"
                                        +"<td><span title='"
                                        + desc.path
                                        + "'>" 
                                        + desc.descriptor_class
                                        + "</span></td></tr>";
                            //console.log("47:"+line);
                            buff+=line;
                        }
                        buff += "</table>";
                        $($(".lp_descriptors_content")[0]).html(buff);
                        
                   }
                   </script>
                   <div class = "lp_descriptors_content" style="border:solid black 1px;padding:2px;margin:3px">
                   </div>                              
                </div> 
                """     
   
class JSRecipeSystem(JSDiv):
    def div(self):
         return """<div class="lp_recipesystem" style="float:left;width:38%">
                   <script type="text/javascript">
                   function populateRecipeSystemDiv(fullmap)
                   {
                        //alert(JSON.stringify(recmap));
                        if (false)
                        {
                            var     recmap = fullmap["recipes"]
                            buff = "<b>Recipies within <tt>"
                                    +fullmap["package"]
                                    +"</tt> Package</b><br/>";
                            for (var recname in recmap)
                            {
                                var rec = recmap[recname];
                                //console.log("js86:"+rec)
                                line =      "<span title='"
                                            + rec.path
                                            + "'>" 
                                            + rec.name 
                                            + "</span><br/>";
                                // console.log("47:"+line);
                                buff+=line;
                            }
                            $($(".lp_recipes_content")[0]).html(buff);
                        }
                        var el = $($(".lp_recipes_content")[0]);
                        el.empty();
                        el.append("<b>Recipes Declared within <tt>"
                                    +fullmap["package"]
                                    +"</tt> Package</b><br/>");
                        el.append(
                                dict2accordian(fullmap["recipe_cat"], el)
                                );
                        // console.log("178:"+JSON.stringify(fullmap["recipe_cat"], null, 3));
                        
                        
                        // PRIMITIVE_SET
                        var prim_cat = fullmap["primitives_cat"];
                        el = $($(".lp_primitives_content")[0]);
                        el.empty();                        
                        var keys = getKeys(prim_cat);
                        
                        buff = "<table>";
                        buff += "<tr><th>Type</th><th>Primitive Set Class</th></tr>"
                        for (var i = 0; i < keys.length; i++)
                        {
                            var key = keys[i];
                            var ps = prim_cat[key];
                            line =      "<tr>"
                                        +"<td>" + key + "</td>"
                                        +"<td><span title='"
                                        + ps.path
                                        + "'>" 
                                        + ps.class
                                        + "</span></td></tr>";
                            //console.log("47:"+line);
                            buff+=line;
                        }
                        buff += "</table>";
                        el.append($(buff));
                   }
                   </script>
                   <div class = "lp_recipes_content" style="border:solid black 1px;padding:2px;margin:3px">
                   </div>
                   <div class = "lp_primitives_content" style="border:solid black 1px;padding:2px;margin:3px">
                   </div>                               
                </div> 
                """      
class JSPackageSelection(JSDiv):
    options = None
        
    def __init__(self, opts):
        self.options = opts

    def div(self):
        ret = """
            <script type="text/javascript">
            function loadPkgContent(url, name)
            {
                //alert("here");
                $.ajax({ url: url,
                        dataType: "json",
                        success: function(data, ts, jq)
                                 {
                                    //alert(JSON.stringify(data));
                                     populateTypesDiv(data);
                                     populateDescriptorsDiv(data);
                                     populateRecipeSystemDiv(data);
                                 }
                        }
                    );
            }            
            </script>
            """ 
        # ret += "<ul>"
        
        for el in self.options:
            ret += """<input type="submit" value="%(name)s" onclick="loadPkgContent('%(url)s', '%(name)s')"></input>
                   """ % {
                          "name" : el["name"], 
                          "url"  : el['url']
                         }
            
        tret = """  <div style="padding:5px;border:solid black 1px;">
                    %s                    
                    </div> <br clear="all"/>                
                """ % ret
        return tret
