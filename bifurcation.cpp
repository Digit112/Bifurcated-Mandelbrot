#include <math.h>
#include <stdio.h>
#include <time.h>

bool d_equ(double a, double b, double epsilon) {
	return abs(a-b) < epsilon;
}

float* mandelbrot(double cr, double ci, float* buf, int buf_n) {
	double zr = 0;
	double zi = 0;
	
	double tr;
	
	float* mem = new float[buf_n*2];
	
	buf[0] = 0;
	
	for (int i = 0; i < buf_n; i++) {
		tr = zr*zr - zi*zi + cr;
		zi = zr * zi * 2 + ci;
		
		zr = tr;
		
		if (zr > 10000 || zi > 10000) {
			delete[] mem;
			return buf;
		}
		
		mem[i*2]   = zr;
		mem[i*2+1] = zi;
		
		for (int j = 0; j < i; j++) {
			if ( d_equ(mem[j*2], mem[i*2], 0.000001) && d_equ(mem[j*2+1], mem[i*2+1], 0.000001) ) {
//			if (mem[j*2] == mem[i*2] && mem[j*2+1] == mem[i*2+1]) {
				buf[0] = i-j;
				for (int k = j; k < i; k++) {
					buf[k-j+1] = mem[k*2];
				}
				delete[] mem;
				return buf;
			}
		}
	}
	
	delete[] mem;
	return buf;
}

int main() {
	printf("Hello Mandelbrot!\n");
	
	FILE* fp = fopen("out.bin", "w+");
	
	int width = 4096;
	int height = 4096;
	
	int iter = 16384;
	
	float* buf = new float[iter];
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	double cr; double ci;
	for (int y = 0; y < height; y++) {
		if (y % 10 == 0) { printf("%.2f%%\n", (float) y/height*100); }
		for (int x = 0; x < width; x++) {
			cr = (double) x/width *3-2;
			ci = (double) y/height*3-1.5;
			
			mandelbrot(cr, ci, buf, iter);
			
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
