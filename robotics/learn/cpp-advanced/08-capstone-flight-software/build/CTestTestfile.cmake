# CMake generated Testfile for 
# Source directory: /home/viku/ai/issueanalyser/learn/cpp-advanced/08-capstone-flight-software
# Build directory: /home/viku/ai/issueanalyser/learn/cpp-advanced/08-capstone-flight-software/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(flight_sw_tests "/home/viku/ai/issueanalyser/learn/cpp-advanced/08-capstone-flight-software/build/flight_sw_tests")
set_tests_properties(flight_sw_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/viku/ai/issueanalyser/learn/cpp-advanced/08-capstone-flight-software/CMakeLists.txt;71;add_test;/home/viku/ai/issueanalyser/learn/cpp-advanced/08-capstone-flight-software/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
