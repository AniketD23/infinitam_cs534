option(WITH_BOOST "Build with Boost support?" ON)

if (WITH_BOOST)
    find_package(Boost COMPONENTS iostreams REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})
endif()
