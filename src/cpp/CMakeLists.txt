CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(PYOPENPOSE)

# setup header
INCLUDE($ENV{CMAKE_COMMON_HEADER_FILE})

# add project directories
ADD_SUBDIRECTORY(pyopenpose)

# setup footer
INCLUDE(FooterConfig)
