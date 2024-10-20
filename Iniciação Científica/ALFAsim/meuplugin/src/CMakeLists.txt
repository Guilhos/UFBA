add_library(meuplugin SHARED meuplugin.cpp hook_specs.h)
target_include_directories(meuplugin PUBLIC ${SDK_INCLUDE_DIR})
install(TARGETS meuplugin EXPORT ${PROJECT_NAME}_export DESTINATION ${ARTIFACTS_DIR})
