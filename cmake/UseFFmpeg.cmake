###################
# UseFFmpeg.cmake #
###################

OPTION(WITH_FFMPEG "Build with FFmpeg support?" OFF)

IF(WITH_FFMPEG)
  pkg_check_modules(libavcodec_illixr REQUIRED libavcodec_illixr)
  pkg_check_modules(libavdevice_illixr REQUIRED libavdevice_illixr)
  pkg_check_modules(libavformat_illixr REQUIRED libavformat_illixr)
  pkg_check_modules(libavutil_illixr REQUIRED libavutil_illixr)
  pkg_check_modules(libswscale_illixr REQUIRED libswscale_illixr)

  set(FFMPEG_LIBRARIES "${libavcodec_illixr_LIBRARIES};${libavdevice_illixr_LIBRARIES};${libavformat_illixr_LIBRARIES};${libavutil_illixr_LIBRARIES};${libswscale_illixr_LIBRARIES}")
  set(FFMPEG_INCLUDE_DIRS "${libavcodec_illixr_INCLUDE_DIRS}")

  INCLUDE_DIRECTORIES(${FFMPEG_INCLUDE_DIRS})
  ADD_DEFINITIONS(-DCOMPILE_WITH_FFMPEG)

ENDIF()
