#pragma once

#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>  /* defines FILENAME_MAX */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <JC/util.h>

using namespace std;

#define uint unsigned int

namespace jc {

	//class OpenCLHost {
	//public:

	//	// ** Attributes **
	//	cl::Device device;
	//	cl::Context context;
	//	cl::CommandQueue queue;
	//	cl::Program program;
	//	cl::Kernel kernel;

	//	// ** Constructor **
	//	OpenCLHost(int PLATFORM_ID, int DEVICE_ID, int argc, char *argv[], string kernel_file, string kernel_name);
	//};
	//// ** Implementation **
	//OpenCLHost::OpenCLHost(int PLATFORM_ID, int DEVICE_ID, int argc, char *argv[], string kernel_file, string kernel_name)
	//	:device(), context(device), queue(context, device, CL_QUEUE_PROFILING_ENABLE), program(buildProgram(kernel_file, context, device)), kernel(program, kernel_name.c_str())
	//{
	//}

const char* readableStatus(cl_int status);
const char* readableCacheType(cl_device_mem_cache_type ct);
const char* readableDeviceType(cl_device_type dt);
void showAllOpenCLDevices();

// ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
// ********** Running and Timing Kernels ************
// ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

string fileToString(const string& file_name) {
    string file_text;

    ifstream file_stream(file_name.c_str());
    if (!file_stream) {
		string file_name_up("..\\");
		file_name_up.append(file_name);
		ifstream file_stream_up(file_name_up.c_str());

		if (!file_stream_up) {
			cout << "Current folder is " << currentFolder() << '\n';
			ostringstream oss;
			oss << "There is no file called " << file_name;
			throw runtime_error(oss.str());
		}
		else {
			file_text.assign(istreambuf_iterator<char>(file_stream_up), istreambuf_iterator<char>());
			return file_text;
		}
    }
    file_text.assign(istreambuf_iterator<char>(file_stream), istreambuf_iterator<char>());
    return file_text;
}

cl::Program stringToProgram(const string& source_code, const cl::Context& context, const cl::Device& device, const char* options = nullptr)
{
	cl::Program program(context, source_code);
	try {
		program.build({device}, options);
    }
    catch (cl::Error& e) {
        string msg;
        program.getBuildInfo<string>(device, CL_PROGRAM_BUILD_LOG, &msg);
        cerr << "Your kernel failed to compile" << endl;
        cerr << "-----------------------------" << endl;
        cerr << msg;
        throw(e);
    }
    return program;
}

cl::Program buildProgram(const string& file_name, const cl::Context& context, const cl::Device& device, const char* options = nullptr)
{
	string source_code = jc::fileToString(file_name);
	return stringToProgram(source_code, context, device, options);
}


// returns run time in nanoseconds
cl_ulong runAndTimeKernel(const cl::Kernel& kernel, const cl::CommandQueue& queue, const cl::NDRange global, const cl::NDRange& local=cl::NullRange)
{
    cl_ulong t1, t2;
    cl::Event evt;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &evt);
    evt.wait();
    evt.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &t1);
    evt.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &t2);

    return t2 - t1;
}
// ** ** ** ** ** ** ** ** ** ** **
// ********** GPU Info ************
// ** ** ** ** ** ** ** ** ** ** **

string deviceName(cl::Device& device) {
	string deviceName;
	device.getInfo<string>(CL_DEVICE_NAME, &deviceName);
	return deviceName;
}
string deviceName(int platformID, int deviceID) {
	vector<cl::Platform> platforms;
	vector<cl::Device> devices;
	cl::Platform::get(&platforms);
	if (platformID >= platforms.size() || platformID < 0)
		return string("");
	platforms[platformID].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (deviceID >= devices.size() || deviceID < 0)
		return string("");
	cl::Device device = devices[deviceID];
	string deviceName;
	device.getInfo<string>(CL_DEVICE_NAME, &deviceName);
	return deviceName;
}
// in MHz
uint clockFrequency(cl::Device& device) {
	cl_uint clockFrequency;
	device.getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY, &clockFrequency);
	return (uint)clockFrequency;
}
// returns local memory size in Bytes
size_t localMemorySize(cl::Device& device) {
	cl_ulong localMemSize;
	device.getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
	return (size_t)localMemSize;
}
// returns number of cores (Compute Units), called Streaming MultiProcessors by Nvidia
uint nbrComputeUnits(cl::Device& device) {
	cl_uint nbrComputeUnits;
	device.getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS, &nbrComputeUnits);
	return (uint)nbrComputeUnits;
}
uint maxWorkGroupSize(cl::Device& device) {
	size_t maxWorkGroupSize;
	device.getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
	return (uint)maxWorkGroupSize;
}
string deviceVendor(cl::Device& device) {
	string vendor;
	device.getInfo(CL_DEVICE_VENDOR, &vendor);
	return vendor;
}
uint warpSize(cl::Device& device) {
	string vendor = deviceVendor(device);
	if (vendor.find("NVIDIA") >= 0)
		return 32;
	else if (vendor.find("AMD") >= 0 || (vendor.find("Advanced")>=0 && vendor.find("Micro") >= 0 && vendor.find("Devices")>=0))
		return 64;
	return 1;
}

// returns 0 if not NVIDIA
// see https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
float nvidiaComputeCability(cl::Device device) {
	cl_uint cc_major, cc_minor;
	try {
		device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, &cc_major);
		device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, &cc_minor);
	}
	catch (...) {
		return 0;
	}
	float cc = (float)cc_major + ((float)cc_minor) / 10.0;
	return cc;
}

string nvidiaArchitecture(float cc) {
	if (cc < 1)
		return "Not nvidia";
	else if (cc < 2)
		return "Tesla";
	else if (cc < 3)
		return "Fermi";
	else if (cc < 4)
		return "Kepler";
	else if (cc < 6)
		return "Maxwell";
	else if (cc < 7)
		return "Pascal";
	else if (cc < 8)
		return "Turing"; // < 8.5 is actually Volta
	else if (cc < 9)
		return "Ampere";
	else
		return "Unknown";
}
string nvidiaArchitecture(cl::Device device) {
	float cc = nvidiaComputeCability(device);
	return nvidiaArchitecture(cc);
}
class OccupancyIterator {
public:
	uint device_localMem, device_maxWorkGroupSize, device_maxNbrWarps, device_warpSize, totalNbrWarps;

	uint concWarpValue = 1, nbrConcurrentWarps=0, actualNbrConcWarps = 0, nbrConcWorkgroups = 0, workGroupSize = 0, workGroupWarps = 0, localMemSize = 0;

	OccupancyIterator(cl::Device& device, int totalNbrWorkItems) {
		device_localMem = localMemorySize(device);
		device_maxWorkGroupSize = maxWorkGroupSize(device);
		device_warpSize = max(warpSize(device), (uint)8); // to limit the iterations
		device_maxNbrWarps = device_maxWorkGroupSize / device_warpSize; 
		totalNbrWarps = totalNbrWorkItems / device_warpSize;
		reset();
		cout << "Occupancy iterator created for device " << deviceName(device) << " from vendor " << deviceVendor(device) << ": local mem=" << device_localMem << "B, maxWorkGroupSize = "<< device_maxWorkGroupSize <<", warp size = " << device_warpSize << ", MAX OCCUPANCY = "<< maxConcWarps() <<" warps = " << (maxConcWarps() * device_warpSize) <<" work items" << endl;
	}
	uint maxConcWarps() {  return (uint)1024 / device_warpSize;  }
	// Fermi: 48 - Kepler, Maxwell, Pascal: 64

	// ** iterator **
	void reset() { concWarpValue = 1;  }
	bool hasNext() { return concWarpValue <= maxConcWarps(); }
	uint next() {
		setOccupancy(concWarpValue);  
		uint current_value = concWarpValue;
		// look for next value
		do { 
			concWarpValue++;
			float nbr_conc_workgroups = ceilf((float)nbrConcurrentWarps / device_maxNbrWarps);
			uint workGroupWarps = (uint)ceilf((float)concWarpValue / nbr_conc_workgroups);
			concWarpValue = workGroupWarps * nbrConcWorkgroups;
		} while (totalNbrWarps % concWarpValue != 0);
		return current_value;
	}
	// ** define the occupancy **
	void setOccupancy(uint nbrConcurrentWarps) {
		this->nbrConcurrentWarps = nbrConcurrentWarps; // TODO actual conc warps
		
		nbrConcWorkgroups = (uint) ceilf((float)nbrConcurrentWarps / device_maxNbrWarps);
		workGroupWarps  = (uint)ceilf((float)nbrConcurrentWarps / nbrConcWorkgroups);
		actualNbrConcWarps = workGroupWarps * nbrConcWorkgroups;
		workGroupSize = workGroupWarps * device_warpSize;
		// JAVA:	localMemSize = Math.max(conf.minimalLocalMem, (int)(device.MAXLOCALMEM - 100) / conf.nbrConcWorkgroups - conf.internalLocalMem);
		localMemSize = (uint)((device_localMem - 128) / nbrConcWorkgroups);
		localMemSize = (uint)ceilf(localMemSize / 4.0f) * 4; // make it a divisor of 4 - TODO: chunk size
		// the chunk size is 256 bytes: 192 x 256 B = 48 KB.
	}
	// ** get the configuration parameters **
	uint getWorkGroupSize() {  return workGroupSize;  }
	uint getNbrConcWorkgroups() {  return nbrConcWorkgroups;  }
	size_t getLocalMemSizeOfWG() {  return localMemSize;  }
};
// ** ** ** ** ** ** ** ** ** ** ** **
// ********** GPU Devices ************
// ** ** ** ** ** ** ** ** ** ** ** **
int numberPlatforms() {
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	return platforms.size();
}
int numberDevices(int platformID) {
	vector<cl::Platform> platforms;
	vector<cl::Device> devices;
	cl::Platform::get(&platforms);
	if (platformID >= platforms.size() || platformID < 0)
		return 0;
	platforms[platformID].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	return devices.size();
}
cl::Device getDevice(int PLATFORM_ID, int DEVICE_ID, bool PRESS_KEY_TO_CLOSE_WINDOW) {

	vector<cl::Platform> platforms;
	vector<cl::Device> devices;
	cl::Platform::get(&platforms);
	if (PLATFORM_ID >= platforms.size()) {
		cerr << "Platform " << PLATFORM_ID << " does not exist on this computer" << endl;
		cerr << "OpenCL platforms & devices on this computer: " << endl;
		jc::showAllOpenCLDevices();
		if (PRESS_KEY_TO_CLOSE_WINDOW) { cout << endl << "Press ENTER to close window..."; char c = cin.get(); }
		exit(-1);
	}

	platforms[PLATFORM_ID].getDevices(CL_DEVICE_TYPE_ALL, &devices); // or CL_DEVICE_TYPE_GPU
	if (DEVICE_ID >= devices.size()) {
		cerr << "Device " << DEVICE_ID << " of platform " << PLATFORM_ID << " does not exist on this computer (#=" << devices.size() << ")" << endl;
		cerr << "OpenCL platforms & devices on this computer: " << endl;
		jc::showAllOpenCLDevices();
		if (PRESS_KEY_TO_CLOSE_WINDOW) { cout << endl << "Press ENTER to close window..."; char c = cin.get(); }
		exit(-1);
	}
	return devices[DEVICE_ID];
}
// find device with name
//for (i = 0; i < num_platforms; ++i) {
//	cl_uint num_devices, j;
//	cl_device_id * device_ids;
//
//	err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
//	device_ids = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
//	err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, num_devices, device_ids, NULL);
//
//	for (j = 0; j < num_devices; ++j) {
//		char * name;
//		size_t name_size;
//		err = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, 0, NULL, &name_size);
//		name = (char*)malloc(sizeof(char)*name_size);
//		err = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, name_size, name, NULL);
//		if (strcmp(name, device_name) == 0) {
//			device_id = device_ids[j];
//		}
//		free(name);
//	}
//	if (device_id == 0) {
//		device_id = device_ids[j];
//	}
//
//	free(device_ids);
//}

void showDevice(const cl::Device& device, int i);

void showPlatform(const cl::Platform& platform, int i, bool showDevices)
{
	string platformName, platformVendor, platformProfile, platformVersion, platformExtensions;
	platform.getInfo(CL_PLATFORM_NAME, &platformName);
	platform.getInfo(CL_PLATFORM_VENDOR, &platformVendor);
	platform.getInfo(CL_PLATFORM_PROFILE, &platformProfile);
	platform.getInfo(CL_PLATFORM_VERSION, &platformVersion);
	platform.getInfo(CL_PLATFORM_EXTENSIONS, &platformExtensions);

	cout << "Platform " << i << endl;
	cout << "    Name:       " << platformName << endl;
	cout << "    Vendor:     " << platformVendor << endl;
	cout << "    Profile:    " << platformProfile << endl;
	cout << "    Version:    " << platformVersion << endl;
	cout << endl;

	if (showDevices) {
		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		for (int i = 0; i < devices.size(); ++i) {
		    showDevice(devices[i], i);
		}
	}
}
void showPlatform(const cl::Platform& platform, int i) {
	showPlatform(platform, i, false);
}
void showPlatformAndDevices(const cl::Platform& platform, int i) {
	showPlatform(platform, i, true);
}
void showDevice(const cl::Device& device, int i)
{
	string deviceName, openCLCVersion, openCLExtensions;
	size_t image2dMaxHeight, image2dMaxWidth;
	size_t image3dMaxDepth, image3dMaxHeight, image3dMaxWidth;
	size_t maxWorkGroupSize, timerResolution;
	
	cl_ulong maxSize, localMemSize;
	cl_uint nbrComputeUnits, nativeVectorWidthFloat, clockFrequency;
	device.getInfo<string>(CL_DEVICE_NAME, &deviceName);
	cl_device_type dt;
	device.getInfo(CL_DEVICE_TYPE, &dt);
	string deviceVendor;
	device.getInfo(CL_DEVICE_VENDOR, &deviceVendor);
	device.getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS, &nbrComputeUnits);
	device.getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &nativeVectorWidthFloat);
	device.getInfo<string>(CL_DEVICE_OPENCL_C_VERSION, &openCLCVersion);
	device.getInfo<string>(CL_DEVICE_EXTENSIONS, &openCLExtensions);
	device.getInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxSize);
	cl_ulong globalMemSize, globalMemCacheSize;
	device.getInfo<cl_ulong>(0x101F, &globalMemSize);  // see cl.h for the code
	device.getInfo<cl_ulong>(0x101E, &globalMemCacheSize);

	// CL_DRIVER_VERSION = JOCLDeviceQuery.getString(device, 0x102D);

	device.getInfo<size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &image2dMaxHeight);
	device.getInfo<size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH, &image2dMaxWidth);
	device.getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH, &image3dMaxDepth);
	device.getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &image3dMaxHeight);
	device.getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH, &image3dMaxWidth);

	device.getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
	device.getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	
	device.getInfo<size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &timerResolution);

	device.getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY, &clockFrequency);

	cl_device_mem_cache_type ct;
	device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &ct);
	
	

	cout << "    Device: " << i << endl;
	cout << "        Name                    : " << deviceName << endl;
	cout << "        Device Type             : " << readableDeviceType(dt) << endl; 
	cout << "        Device Type             : " << deviceVendor << endl;
	cout << "        OpenCL C Version        : " << openCLCVersion << endl;
	cout << "        #Compute Units (cores)  : " << nbrComputeUnits << endl;
	cout << "        Native vector width     : " << nativeVectorWidthFloat << endl;
	cout << "        2D Image limits         : " << image2dMaxHeight << "x" << image2dMaxWidth << endl;
	cout << "        3D Image limits         : " << image3dMaxDepth << "x" << image3dMaxHeight << "x" << image2dMaxWidth << endl;
	cout << "        Global memory size [MB] : " << globalMemSize / (1024 * 1024) << endl;
	cout << "        Global memory cache size [KB] : " << globalMemCacheSize / 1024 << endl;
	cout << "        Local memory size [KB]  : " << localMemSize / 1024 << endl;
	cout << "        Maximum buffer size [MB]: " << maxSize / (1024 * 1024) << endl;
	cout << "        Maximum workgroup size  : " << maxWorkGroupSize << endl;
	std::vector<::size_t> maxWorkItemsSizes;
	maxWorkItemsSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
//	cout << "        Maximum workitems in grp: : " << maxWorkItemsSizes[0] << ", "<< maxWorkItemsSizes[1] << ", " << maxWorkItemsSizes[2] <<  endl;
	cout << "        Timer resolution        : " << timerResolution << endl;
	cout << "        Clock frequency  [MHz]  : " << clockFrequency << endl;
	cout << "        Cache Type              : " << readableCacheType(ct) << endl;

	try {
		// FOR NVIDIA GPUs
		// see https://www.khronos.org/registry/OpenCL/extensions/nv/cl_nv_device_attribute_query.txt
		cl_uint cc_major, cc_minor, registers_per_block, warp_size;
		cl_bool has_integrated_memory;
		device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, &cc_major);
		device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, &cc_minor);
		device.getInfo(CL_DEVICE_REGISTERS_PER_BLOCK_NV, &registers_per_block);
		device.getInfo(CL_DEVICE_WARP_SIZE_NV, &warp_size);
		device.getInfo(CL_DEVICE_INTEGRATED_MEMORY_NV, &has_integrated_memory);
		cout << "        Compute capability      : " << cc_major << "." << cc_minor << endl;
		cout << "        Nvidia architecture     : " << nvidiaArchitecture((float)cc_major) << endl;
		cout << "        Registers per compute unit (32-bit) : " << registers_per_block << endl;
		cout << "        Warp size               : " << warp_size << endl;
		cout << "        device is integrated with the memory subsystem  : " << has_integrated_memory << endl;
	}
	catch (...) {
		// ignore
	}
	cout << endl;
}
// Detailed Info
void showAllOpenCLDevices()
{
	try {
		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::Platform::get(&platforms);

		for (int i = 0; i < platforms.size(); ++i) {
			showPlatform(platforms[i], i, true);
		}
	}
	catch (cl::Error &e) {
		cerr << e.what() << ":" << readableStatus(e.err());
		return;
	}
	catch (exception& e) {
		cerr << e.what();
		return;
	}
	catch (...) {
		cerr << "Unforeseen error";
		return;
	}

//cout << endl << "Press ENTER to close window...";
//char c = cin.get();
}


void showListOfGPUs() {
	try {

		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::Platform::get(&platforms);

		for (int i = 0; i < platforms.size(); ++i) {
			cl::Platform platform = platforms[i];
			string platformName, platformVendor;
			platform.getInfo(CL_PLATFORM_NAME, &platformName);
			platform.getInfo(CL_PLATFORM_VENDOR, &platformVendor);

			vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			for (int j = 0; j < devices.size(); ++j) {
				cl::Device device = devices[j];
				string deviceName, openCLCVersion, openCLExtensions;
				
				size_t maxWorkGroupSize, timerResolution;
				cl_ulong maxSize, localMemSize;
				cl_uint nativeVectorWidthFloat, clockFrequency, nbrComputeUnits;
				device.getInfo<string>(CL_DEVICE_NAME, &deviceName);
			//	device.getInfo<string>(CL_DEVICE_OPENCL_C_VERSION, &openCLCVersion);
			//	device.getInfo<string>(CL_DEVICE_EXTENSIONS, &openCLExtensions);
			//	device.getInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxSize);
				device.getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS, &nbrComputeUnits);
			//	device.getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
				device.getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &nativeVectorWidthFloat);
			//	device.getInfo<size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &timerResolution);
				device.getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY, &clockFrequency);

				cout << " - " << deviceName << " on " << platformName<<": " << nbrComputeUnits <<" cores " << clockFrequency <<
					"MHz " << " vector width="<< nativeVectorWidthFloat <<endl;
				//cout << "        OpenCL C Version        :" << openCLCVersion << endl;
				//cout << "        Maximum workgroup size  :" << maxWorkGroupSize << endl;
				//cout << "        Native vector width     :" << nativeVectorWidthFloat << endl;
				//cout << "        Timer resolution        :" << timerResolution << endl;
				//cout << "        Clock frequency         :" << clockFrequency << endl;
				//cout << endl;
			}
		}
	}
	catch (cl::Error &e) {
		cerr << e.what() << ":" << readableStatus(e.err());
		return;
	}
	catch (exception& e) {
		cerr << e.what();
		return;
	}
	catch (...) {
		cerr << "Unforeseen error";
		return;
	}
}


// ** ** ** ** ** ** ** ** ** ** ** ** ** *
// ********** OpenCL constants ************
// ** ** ** ** ** ** ** ** ** ** ** ** ** *

const char *readableStatus(cl_int status)
{
	switch (status) {
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
#ifndef CL_VERSION_1_0
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifndef CL_VERSION_1_0
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
#endif
	default:
		return "CL_UNKNOWN_CODE";
	}
}
const char* readableCacheType(cl_device_mem_cache_type ct)
{
	switch (ct) {
	case CL_READ_ONLY_CACHE:
		return "CL_READ_ONLY_CACHE";
	case CL_READ_WRITE_CACHE:
		return "CL_READ_WRITE_CACHE";
	case CL_NONE:
		return "CL_NONE";
	default:
		return "CL_UNK_CACHE";
	}
}

const char* readableDeviceType(cl_device_type dt)
{
	switch (dt) {
	case CL_DEVICE_TYPE_CPU:
		return "CL_DEVICE_TYPE_CPU";
	case CL_DEVICE_TYPE_GPU:
		return "CL_DEVICE_TYPE_GPU";
	default:
		return "CL_DEVICE_TYPE_UNK";
	}
}
} // namespace JC

