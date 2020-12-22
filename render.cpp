#include <math.h>
#include <stdio.h>
#include <time.h>

#include <CL/cl.h>

#include "vec.hpp"

bool d_equ(double a, double b, double epsilon) {
	return abs(a-b) < epsilon;
}

int main() {
	const float d2r = 3.14159/180;
	
	// Initial values
	size_t width = 3072;
	size_t height = 3072;
	
	int data_width = 1000;
	int data_height = 1000;
	
	int frames = 200;
	int frame_s = 136;
	int frame_e = 200;
	
	// Number of sub-images to divide each edge of the image over
	// If the output is 16x16 and sub_img is 2, the work will be divided into 4 images each os size 8x8
	// Ensure that both width and height are easily divisible into sub_img
	int sub_img = 6;
	
	size_t sub_img_edge_x = width  / sub_img;
	size_t sub_img_edge_y = height / sub_img;
	
	vecd3 cam_p_i(-3, 0, 1);
	quaternion cam_r_i(vecd3(0, 1, 0), 20*d2r);
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	printf("Loading...\n");
	
	// Load data
	FILE* fp = fopen("out.bin", "r");
	
	fseek(fp, 0, SEEK_END);
	int raw_size = ftell(fp);
	rewind(fp);
	
	int rawp = data_width*data_height;
	int lkpp = 0;
	
	int read;
	float* raw_data = new float[data_width*data_height+raw_size];
	for (int i = 0; i < data_width; i++) {
		float size;
		float level;
		for (int j = 0; j < data_height; j++) {
			read = fread(&size, 4, 1, fp);
			
			raw_data[lkpp] = rawp;
			lkpp++;
			
			raw_data[rawp] = size;
			rawp++;
			
			for (int k = 1; k <= size; k++) {
				read = fread(&level, 4, 1, fp);
				raw_data[rawp] = level;
				rawp++;
			}
		}
	}
	
	fclose(fp);
	
	printf("Getting Device...\n");
	
	// Initialize OpenCL
	int errcd;
	
	cl_platform_id platform;
	errcd = clGetPlatformIDs(1, &platform, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cl_device_id device;
	errcd = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	char* cbuf = new char[1024];
	errcd = clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, cbuf, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	printf("%s\n", cbuf);
	delete[] cbuf;
	
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cbuf = new char[8192];
	fp = fopen((char*) "render.cl", "r");
	size_t source_size;
	source_size = fread(cbuf, 1, 8192, fp);
	cbuf[source_size] = '\0';
	fclose(fp);
	printf("Read %ld bytes of code into cbuf.\n", source_size);
	
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char**) &cbuf, &source_size, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	delete[] cbuf;
	
	errcd = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	if (errcd == CL_BUILD_PROGRAM_FAILURE || errcd == -9999) {
		size_t log_size;
		char* log;
		
		errcd = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		log = new char[log_size];
		errcd = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		printf("%s\n", log);
		delete[] log;
	}
	
	cl_kernel render;
	render = clCreateKernel(program, (char*) "render", &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cl_command_queue queue;

#ifdef CL_VERSION_2_0
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &errcd);
#else
	queue = clCreateCommandQueue(context, device, 0, &errcd);
#endif
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	// Move raw_data to GPU global memory
	cl_mem cldata = clCreateBuffer(context, CL_MEM_READ_ONLY, (data_width*data_height + raw_size)*sizeof(float), NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	// Send data to buffer
	clEnqueueWriteBuffer(queue, cldata, CL_TRUE, 0, (data_width*data_height + raw_size)*sizeof(float), raw_data, 0, NULL, NULL);
	
	delete[] raw_data;
	
	cl_mem imbuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height, NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cl_image_format imform = {CL_R, CL_UNSIGNED_INT8};
	cl_image_desc imdesc = {CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, imbuf};
	
	cl_mem out = clCreateImage(context, CL_MEM_WRITE_ONLY, &imform, &imdesc, NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	errcd = clSetKernelArg(render, 0, sizeof(int), &width);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	errcd = clSetKernelArg(render, 1, sizeof(int), &height);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	errcd = clSetKernelArg(render, 2, sizeof(cl_mem), &out);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	errcd = clSetKernelArg(render, 3, sizeof(cl_mem), &cldata);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	size_t* global_work_size = new size_t[2];
	global_work_size[0] = sub_img_edge_x;
	global_work_size[1] = sub_img_edge_y;
	
	size_t* global_work_offset = new size_t[2];
	global_work_offset[0] = 0;
	global_work_offset[1] = 0;
	
	size_t* origin = new size_t[3];
	size_t* region = new size_t[3];
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
		
	unsigned char* imdata = new unsigned char[width*height];
	
	double start_f; double end_f;
	char* fn = new char[32];
	for (int f = frame_s; f < frame_e; f++) {
		printf("Frame %d...", f);
		fflush(stdout);
		
		start_f = (double) clock() / CLOCKS_PER_SEC;
		
		float rot_theta = 3.14159*2 * f / frames;
		
		quaternion orbit( vecd3(0, 0, 1), rot_theta );
		
		vecd3 cam_p = orbit.apply(cam_p_i) - vecd3(0.5, 0, 0);
		quaternion cam_r = quaternion::hamilton(orbit, cam_r_i);
	
		errcd = clSetKernelArg(render, 4, sizeof(double), &cam_p.x);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		errcd = clSetKernelArg(render, 5, sizeof(double), &cam_p.y);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		errcd = clSetKernelArg(render, 6, sizeof(double), &cam_p.z);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		
		errcd = clSetKernelArg(render, 7, sizeof(double), &cam_r.x);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		errcd = clSetKernelArg(render, 8, sizeof(double), &cam_r.y);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		errcd = clSetKernelArg(render, 9, sizeof(double), &cam_r.z);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		errcd = clSetKernelArg(render, 10,sizeof(double), &cam_r.w);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		
		// Work is divided into sub_img^2 smaller enqueue's to prevent GPU lag
		for (int x = 0; x < sub_img; x++) {
			for (int y = 0; y < sub_img; y++) {
				global_work_offset[0] = sub_img_edge_x * x;
				global_work_offset[1] = sub_img_edge_y * y;
				
				clEnqueueNDRangeKernel(queue, render, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL);
				
				clFinish(queue);
			}
		}
		
		clEnqueueReadImage(queue, out, CL_TRUE, origin, region, 0, 0, imdata, 0, NULL, NULL);
		
		sprintf(fn, "out/%d.bin", f);
		fp = fopen(fn, "w+");
		
		fwrite(imdata, 1, width*height, fp);
		
		fclose(fp);
		
		end_f = (double) clock() / CLOCKS_PER_SEC;
		
		printf(" Completed in %.2fs\n", end_f-start_f);
	}
	
	delete[] fn;
	
	delete[] imdata;
	
	delete[] global_work_size;
	delete[] global_work_offset;
	delete[] origin;
	delete[] region;
	
	double end = (double) clock() / CLOCKS_PER_SEC;
	
	printf("Complete in %.2fs\n", end-start);
	
	return 0;
}
