# #
# #                                                                  gemini_python
# #
# #                                                           primitives_upload.py
# # ------------------------------------------------------------------------------
# import copy

# import astrodata, gemini_instruments

# from gempy.gemini import gemini_tools as gt
# from recipe_system.utils.decorators import parameter_override

# from geminidr import PrimitivesBASE, save_cache, stkindfile
# from geminidr.core import parameters_bookkeeping, parameters_upload
# from os.path import basename

# import requests
# import urllib


# # TODO fix me
# UPLOADPROC    = "http://localhost:8090/upload_file" #  UPLOADURL_DICT["UPLOADPROCCAL"]
# #UPLOADPROC    = "http://hbffits-lv4.hi.gemini.edu/upload_file" #  UPLOADURL_DICT["UPLOADPROCCAL"]
# UPLOADCOOKIE  = "have_a_cookie"  # UPLOADURL_DICT["UPLOADCOOKIE"]


# # ------------------------------------------------------------------------------
# @parameter_override
# class Upload(PrimitivesBASE):
#     """
#     This is the class containing all of the preprocessing primitives
#     for the Bookkeeping level of the type hierarchy tree. It inherits all
#     the primitives from the level above
#     """
#     tagset = None

#     def __init__(self, adinputs, **kwargs):
#         super(Upload, self).__init__(adinputs, **kwargs)
#         self._param_update(parameters_upload)

#     def uploadFiles(self, adinputs=None, **params):
#         """
#         A primitive that may be called by a recipe at any stage to
#         upload the outputs to the on-site FITSStorage server.
#         """
#         log = self.log

#         # if we are not configured to do uploads, just exit
#         if not self.upload:
#             return adinputs

#         mode = None
#         if 'ql' in self.mode:
#             mode = 'ql'
#         if 'sq' in self.mode:
#             mode = 'sq'
#         if mode is None:
#             log.warn("Unable to determine mode for upload, skipping")
#             return adinputs

#         # save filenames so we can restore them after
#         filenames = [ad.filename for ad in adinputs]

#         for ad in adinputs:
#             ad.phu.set('PROCSCI', mode)
#             ad.update_filename(suffix="_%s" % mode)
#             ad.write()
#             filename = ad.filename
#             fn = basename(filename)
#             url = "/".join((UPLOADPROC, fn))
#             postdata = open(filename, 'rb').read()
#             try:
#                 rq = urllib.request.Request(url)
#                 rq.add_header('Content-Length', '%d' % len(postdata))
#                 rq.add_header('Content-Type', 'application/octet-stream')
#                 rq.add_header('Cookie', 'gemini_fits_upload_auth=%s' % UPLOADCOOKIE)
#                 u = urllib.request.urlopen(rq, postdata)
#                 response = u.read()
#             except urllib.error.HTTPError as error:
#                 log.error(str(error))
#                 raise

#             # Finally, write the file to the name that was decided upon
#             log.stdinfo("Uploaded file {}".format(filename))

#         # restore filenames
#         for (filename, ad) in zip(filenames, adinputs):
#             ad.filename = filename

#         return adinputs
