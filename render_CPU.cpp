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
	int width = 128;
	int height = 128;
	
	int data_width = 1000;
	int data_height = 1000;
	
	int frames = 1;
	
	float ray_jump = 0.001;
	float ray_thresh = 0.01;
	int ray_max = 5000;
	
	vecd3 cam_p_i(-3, 0, 1);
	quaternion cam_r_i(vecd3(0, 1, 0), 20*d2r);
	
	float theta = 3.14159/3;
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	// Load data
	int read;
	FILE* fp = fopen("out.bin", "r");
	float*** data = new float**[data_width];
	for (int i = 0; i < data_width; i++) {
		data[i] = new float*[data_height];
		
		float size;
		float level;
		for (int j = 0; j < data_height; j++) {
			read = fread(&size, 4, 1, fp);
			
			data[i][j] = new float[(int) size+1];
			data[i][j][0] = size;
			for (int k = 1; k <= size; k++) {
				read = fread(&level, 4, 1, fp);
				data[i][j][k] = level;
			}
		}
	}
	
	fclose(fp);
	
	float f_start; float f_end;
	float rot_theta;
	unsigned char fa = 0; unsigned char t;
	float xt; float yt;
	
	vecd3 cam_p;
	quaternion cam_r;
	
	vecd3 ray_pos; vecd3 ray_dir;
	
	int xi; int yi;
	bool hit;
	
	int x; int y; int i; int j;
	
	char* fn = new char[32];
	for (int f = 0; f < frames; f++) {
		printf("Frame %d... ", f);
		fflush(stdout);
		
		f_start = clock() / CLOCKS_PER_SEC;
		
		sprintf(fn, "out/%d.bin", f);
		fp = fopen(fn, "w+");
		
		rot_theta = 3.14159*2 * f / frames;
		
		quaternion orbit( vecd3(0, 0, 1), rot_theta );
		
		cam_p = orbit.apply(cam_p_i) - vecd3(0.5, 0, 0);
		cam_r = quaternion::hamilton(orbit, cam_r_i);
		
		for (y = 0; y < height; y++) {
			yt = (float) y/height * theta - theta/2;
			for (x = 0; x < width; x++) {
				xt = (float) x/width * theta - theta/2;
				
				ray_pos = cam_p;
				
				ray_dir.x = 1;
				ray_dir.y = tan(xt);
				ray_dir.z = tan(yt);
				
				ray_dir = ray_dir.normalize(ray_jump);
				
				ray_dir = cam_r.apply(ray_dir);
				
				hit = false;
				for (i = 0; true; i++) {
					if ( (ray_pos.x < -2 && ray_dir.x < 0) || (ray_pos.x > 1 && ray_dir.x > 0) || (ray_pos.y < -1.5 && ray_dir.y < 0) || (ray_pos.y > 1.5 && ray_dir.y > 0) || i == ray_max ) {
						fwrite(&fa, 1, 1, fp);
						break;
					}
					
					if (ray_pos.x > -2 && ray_pos.x < 1 && ray_pos.y > -1.5 && ray_pos.y < 1.5) {
						xi = (int) ((ray_pos.x + 2.0)/3*data_width);
						yi = (int) ((ray_pos.y + 1.5)/3*data_height);
						
						for (j = 1; j <= data[xi][yi][0]; j++) {
							if (d_equ(ray_pos.z, data[xi][yi][j], ray_thresh)) {
								t = (unsigned char) ((data[xi][yi][j]+2)/4 * 255);
								fwrite(&t, 1, 1, fp);
								hit = true;
								break;
							}
						}
						
						if (hit) {break;}
					}
					
					ray_pos = ray_pos + ray_dir;
				}
			}
		}
		
		fclose(fp);
		
		f_end = clock() / CLOCKS_PER_SEC;
		
		printf("Completed in %.2f seconds.\n", f_end-f_start);
	}
	
	delete[] fn;
	
	for (int i = 0; i < data_width; i++) {
		for (int j = 0; j < data_height; j++) {
			delete[] data[i][j];
		}
		delete[] data[i];
	}
	delete[] data;
	
	double end = (double) clock() / CLOCKS_PER_SEC;
	
	printf("Complete in %.2fs\n", end-start);
	
	return 0;
}
