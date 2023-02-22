#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Check that two doubles are equal to within a small error.
__device__ bool d_equ(double a, double b, double epsilon) {
	return fabs(a-b) < epsilon;
}

// Iterate over the given point until buf_n iterations are reached or until the point Z falls into a loop or escapes.
// If the point falls into a loop or converges to a point (A "loop of length 1") then copy the real components visited
// during the loop into buf and return. buf_n should be the length of buf in floats, so the length in bytes should be buf_n*4.
__global__ void mandelbrot(float* buf, int iter) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	int width = blockDim.x * gridDim.x;
	int height = blockDim.y * gridDim.y;
	
	int ind = (y*width + x) * 200;
	
	int is_test = (x == 300 && y == 200);
	
//	printf("%d, %d, %d, %d, %d\n", x, y, width, height, ind);
	
	double zr = 0;
	double zi = 0;
	
	double cr = (double) x / width * 3 - 2;
	double ci = (double) y / height * 3 - 1.5;
	
	double tr;
	
	int next = 0;
	
	float mem[398];
	
	buf[0] = 0;
	
	int rcrd_len = 0;
	
	// Iterate the Mandelbrot set for the given point.
	for (int i = 0; i < iter; i++) {
		tr = zr*zr - zi*zi + cr;
		zi = zr * zi * 2 + ci;
		
		zr = tr;
		
		// If this point escapes, return the buffer without modifying it.
		if (zr > 10 || zi > 10) {
			return;
		}
		
		if (is_test) {
			printf("Recording %.2f, %.2f\n", zr, zi);
		}
		
		// Start recording when half the iterations have been finished.
		if (i > iter / 2)  {
			// Set the relevant locations in memory to hold this step of the iteration.
			mem[next  ] = zr;
			mem[next+1] = zi;
			
			// Iterate over the 200 previous points.
			int jm;
			for (int j = 0; j < rcrd_len && j < 198; j++) {
				jm = (next - 2 - j*2 + 398) % 398;
				
				// Check if this zr, zi is very close to a previous iteration. If so, it's probably in a loop.
				if ( d_equ(mem[jm], zr, 0.000001) && d_equ(mem[jm+1], zi, 0.000001) ) {
					// Copy the real component of every point in the passed buffer and return. The first component of the buffer is the length of the loop.
					buf[ind] = j+1;
					if (is_test) {
						printf("(%d, %d) -> (%.2f, %.2f) Found cycle w/ prd. %d on iter. %d. Between %d, %d\n", x, y, cr, ci, j+1, i, jm, next);
					}
					for (int k = 0; k <= j; k++) {
						jm = (next - k*2 + 398) % 398;
						if (is_test) {
							printf("Writing %.2f to sample.\n", mem[jm]);
						}
						buf[ind+k+1] = mem[jm];
					}
					return;
				}
			}
			
			next = (next + 2) % 398;
			rcrd_len++;
		}
	}
	
	return;
}

int main() {
	printf("Hello Mandelbrot!\n");
	
	int width = 2048;
	int height = 2048;
	
	int iter = 8000;
	
	// Get settings
	// FILE* fin = fopen("conf.txt", "r");
	// char conf[128];
	// for (int i = 0; i < 1000; i++) {
		// fgets(conf, 128, fin);
		
		// if (strncmp("sample_width ", conf, 13) == 0) {
			// width = atoi(conf + 13);
		// }
		// else if (strncmp("sample_height ", conf, 14) == 0) {
			// height = atoi(conf + 14);
		// }
		// else if (strncmp("sample_iters ", conf, 13) == 0) {
			// iter = atoi(conf + 13);
		// }
	// }
	
	// fclose(fin);
	
	FILE* fp = fopen("out.bin", "wb+");
	
	float* buf;
	printf("Buffer Size: %.2fGB\n", (float) (sizeof(float) * 200 * width * height) / 1024 / 1024 / 1024);
	cudaError_t err = cudaMallocManaged(&buf, sizeof(float) * 200 * width * height);
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
	
	for (int i = 0; i < 200 * width * height; i++) {
		buf[i] = 0;
	}
	
	dim3 block(32, 32);
	dim3 grid(width/32, height/32);
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	mandelbrot<<<grid, block>>>(buf, iter);
	
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
	
	// For each pixel, iterate over that point and write the real components of the loop it falls into to file. Each loop is preceded by the length of the loop. They are encoded row-first.
	int num_written = 0;
	int wri_len;
	for (int y = 0; y < height; y++) {
		if (y % 10 == 0) { printf("%.2f%%\n", (float) y/height*100); }
		for (int x = 0; x < width; x++) {
			int ind = (y*width + x)*200;

			wri_len = fwrite(buf + ind, sizeof(float), (int) (buf[ind]+1), fp);
			if (wri_len != buf[ind]+1) {
				printf("Error writing %.2f float(s) from %d. Sample (%d, %d)\n", buf[ind]+1, ind, x, y);
			}
			else {
				num_written++;
			}
		}
	}
	printf("Wrote %d samples.\n", num_written);
	
	double end = (double) clock() / CLOCKS_PER_SEC;
	
	double sec = end-start;
	int min = (int) (sec/60);
	sec = fmod(sec, 60);
	
	printf("Complete in %dm, %.2fs\n", min, sec);
	
	fclose(fp);
	
	delete[] buf;
	return 0;
}
