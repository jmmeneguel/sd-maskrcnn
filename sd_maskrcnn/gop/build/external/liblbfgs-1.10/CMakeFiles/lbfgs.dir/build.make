# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build

# Include any dependencies generated for this target.
include external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/depend.make

# Include the progress variables for this target.
include external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/progress.make

# Include the compile flags for this target's objects.
include external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/flags.make

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/flags.make
external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o: ../external/liblbfgs-1.10/lib/lbfgs.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o"
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/lbfgs.dir/lib/lbfgs.c.o   -c /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/external/liblbfgs-1.10/lib/lbfgs.c

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/lbfgs.dir/lib/lbfgs.c.i"
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/external/liblbfgs-1.10/lib/lbfgs.c > CMakeFiles/lbfgs.dir/lib/lbfgs.c.i

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/lbfgs.dir/lib/lbfgs.c.s"
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/external/liblbfgs-1.10/lib/lbfgs.c -o CMakeFiles/lbfgs.dir/lib/lbfgs.c.s

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.requires:

.PHONY : external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.requires

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.provides: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.requires
	$(MAKE) -f external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/build.make external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.provides.build
.PHONY : external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.provides

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.provides.build: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o


# Object files for target lbfgs
lbfgs_OBJECTS = \
"CMakeFiles/lbfgs.dir/lib/lbfgs.c.o"

# External object files for target lbfgs
lbfgs_EXTERNAL_OBJECTS =

external/liblbfgs-1.10/liblbfgs.a: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o
external/liblbfgs-1.10/liblbfgs.a: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/build.make
external/liblbfgs-1.10/liblbfgs.a: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library liblbfgs.a"
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 && $(CMAKE_COMMAND) -P CMakeFiles/lbfgs.dir/cmake_clean_target.cmake
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lbfgs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/build: external/liblbfgs-1.10/liblbfgs.a

.PHONY : external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/build

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/requires: external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/lib/lbfgs.c.o.requires

.PHONY : external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/requires

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/clean:
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 && $(CMAKE_COMMAND) -P CMakeFiles/lbfgs.dir/cmake_clean.cmake
.PHONY : external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/clean

external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/depend:
	cd /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/external/liblbfgs-1.10 /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10 /home/mjd3/working/depthseg/sd-maskrcnn/sd_maskrcnn/gop/build/external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/liblbfgs-1.10/CMakeFiles/lbfgs.dir/depend
