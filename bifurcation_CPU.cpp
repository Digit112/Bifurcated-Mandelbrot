#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// Check that two doubles are equal to within a small error.
bool d_equ(double a, double b, double epsilon) {
	return abs(a-b) < epsilon;
}

// Iterate over the given point until buf_n iterations are reached or until the point Z falls into a loop or escapes.
// If the point falls into a loop or converges to a point (A "loop of length 1") then copy the real components visited
// during the loop into buf and return. buf_n should be the length of buf in floats, so the length in bytes should be buf_n*4.
float* mandelbrot(double cr, double ci, float* buf, int buf_n, int iters) {
	double zr = 0;
	double zi = 0;
	
	double tr;
	
	float* mem = new float[buf_n*2];
	
	buf[0] = 0;
	
	// Iterate the Mandelbrot set for the given point.
	for (int i = 0; i < iters; i++) {
		tr = zr*zr - zi*zi + cr;
		zi = zr * zi * 2 + ci;
		
		zr = tr;
		
		// If this point escapes, return the buffer without modifying it.
		if (zr > 10 || zi > 10) {
			delete[] mem;
			return buf;
		}
		
		int ind = i - (iters - buf_n);
		
		if (ind >= 0) {
			// Set the relevant locations in memory to hold this step of the iteration.
			mem[ind*2]   = zr;
			mem[ind*2+1] = zi;
			
			// Iterate over all previous points
			for (int j = ind-1; j >= 0; j--) {
				// Check if this zr, zi is very close to a previous iteration. If so, it's probably  in a loop.
				if ( d_equ(mem[j*2], zr, 0.000001) && d_equ(mem[j*2+1], zi, 0.000001) ) {
					
					// Copy the real component of every point in the passed buffer and return. The first component of the buffer is the length of the loop.
					buf[0] = ind-j;
					for (int k = j; k < ind; k++) {
						buf[k-j+1] = mem[(ind-k)*2];
					}
					
					delete[] mem;
					return buf;
				}
			}
		}
	}
	
	delete[] mem;
	return buf;
}

int main() {
	printf("Hello Mandelbrot!\n");
	
	int width = 512;
	int height = 512;
	
	int buf_n = 1400;
	int iter = 4096;
	
	// Get settings
	FILE* fin = fopen("conf.txt", "r");
	char conf[128];
	for (int i = 0; i < 1000; i++) {
		fgets(conf, 128, fin);
		
		if (strncmp("sample_width ", conf, 13) == 0) {
			width = atoi(conf + 13);
		}
		else if (strncmp("sample_height ", conf, 14) == 0) {
			height = atoi(conf + 14);
		}
		else if (strncmp("sample_iters ", conf, 13) == 0) {
			iter = atoi(conf + 13);
		}
	}
	
	fclose(fin);
	
	FILE* fp = fopen("out.bin", "wb+");
	
	float* buf = new float[buf_n];
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	// For each pixel, iterate over that point and write the real components of the loop it falls into to file. Each loop is preceded by the length of the loop. They are encoded row-first.
	double cr; double ci;
	for (int y = 0; y < height / 2; y++) {
		if (y % 10 == 0) { printf("%.2f%%\n", (float) y/height*200); }
		for (int x = 0; x < width; x++) {
			cr = (double) x/width *3-2;
			ci = (double) y/(height/2 - 1)*1.5-1.5;
			
			mandelbrot(cr, ci, buf, buf_n, iter);
			
			fwrite(buf, sizeof(float), buf[0]+1, fp);
		}
	}
	
	double end = (double) clock() / CLOCKS_PER_SEC;
	
	double sec = end-start;
	int min = (int) (sec/60);
	sec = fmod(sec, 60);
	
	printf("Complete in %dm, %.2fs\n", min, sec);
	
	fclose(fp);
	
	delete[] buf;
	return 0;
}
