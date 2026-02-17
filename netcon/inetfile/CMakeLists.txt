set(HEADERS
  httpclient.h
)

set(CPPS
  httpclient.cpp
)

add_library(inetfile STATIC ${HEADERS} ${CPPS})
target_link_libraries(inetfile PRIVATE
  ddio
)
target_link_libraries(inetfile PUBLIC
  httplib # For some reason, linking it privately causes issues at runtime
  SDL2::SDL2
)
target_include_directories(inetfile PUBLIC
  $<BUILD_INTERFACE:
    ${PROJECT_SOURCE_DIR}/netcon/inetfile
  >
  mem
)
target_include_directories(inetfile PRIVATE ${SDL2_INCLUDE_DIRS})

