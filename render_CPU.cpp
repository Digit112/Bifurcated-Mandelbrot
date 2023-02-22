#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "color.hpp"
#include "vec.hpp"

bool d_equ(double a, double b, double epsilon) {
	return abs(a-b) < epsilon;
}

int main() {
	const float d2r = 3.14159/180;
	
	// Initial values
	int sample_width = 1024;
	int sample_height = 1024;
	
	int frames = 48;
	
	int width = 256;
	int height = 256;
	
	// Get settings
	FILE* fin = fopen("conf.txt", "r");
	char conf[128];
	for (int i = 0; i < 1000; i++) {
		fgets(conf, 128, fin);
		
		if (strncmp("sample_width ", conf, 13) == 0) {
			sample_width = atoi(conf + 13);
		}
		else if (strncmp("sample_height ", conf, 14) == 0) {
			sample_height = atoi(conf + 14);
		}
		else if (strncmp("frames ", conf, 7) == 0) {
			frames = atoi(conf + 7);
		}
		else if (strncmp("output_width ", conf, 13) == 0) {
			width = atoi(conf + 13);
		}
		else if (strncmp("output_height ", conf, 14) == 0) {
			height = atoi(conf + 14);
		}
	}
	
	float ray_thresh = 0.003;
	int ray_max = 32000;
	
	// Create the buffer to hold the image to write.
	uint8_t* img = new uint8_t[width*height*3];
	char hdr[64];
	
	vecd3 cam_p_i(-3, 0, 1);
	quaternion cam_r_i(vecd3(0, 1, 0), 20*d2r);
	
	float theta = 3.14159/3;
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	// Load data
	printf("Constructing Sample Grid...\n");
	float*** data = new float**[sample_width];
	for (int i = 0; i < sample_width; i++) {
		data[i] = new float*[sample_height / 2];
	}
		
	printf("Loading...\n");
	int read;
	FILE* fp = fopen("out.bin", "rb");
	for (int j = 0; j < sample_height / 2; j++) {
		if (j % 20 == 0) {
			printf("%.2f%%\n", (float) j / sample_height * 200);
		}
		
		float size;
		float level;
		for (int i = 0; i < sample_width; i++) {
			// Read the size of this loop.
			read = fread(&size, 4, 1, fp);
			
			// Create a new array for this loop. The first item is the length of the loop.
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
	float xt; float yt;
	
	vecd3 cam_p;
	quaternion cam_r;
	
	vecd3 ray_pos; vecd3 ray_dir;
	
	int xi; int yi;
	bool hit;
	
	int x; int y; int i; int j;
	
	char* fn = new char[32];
	for (int f = 0; f < frames; f++) {
		printf("Frame %d... \n", f);
		fflush(stdout);
		
		f_start = clock() / CLOCKS_PER_SEC;
		
		rot_theta = 3.14159*2 * f / frames;
		
		quaternion orbit( vecd3(0, 0, 1), rot_theta );
		
		cam_p = orbit.apply(cam_p_i) - vecd3(0.5, 0, 0);
		cam_r = quaternion::hamilton(orbit, cam_r_i);
		
		for (y = 0; y < height; y++) {
			yt = (float) y/height * theta - theta/2;
			if (y % 20 == 0) {
				printf("%d\n", y);
			}
			
			for (x = 0; x < width; x++) {
				xt = (float) x/width * theta - theta/2;
				
				int ind = ((height - y - 1)*width + x)*3;
				
				// Create a ray at this camera moving in a direction based on the pixel this ray is coming from.
				ray_pos = cam_p;
				
				ray_dir.x = 1;
				ray_dir.y = tan(xt);
				ray_dir.z = tan(yt);
				
				ray_dir = ray_dir.normalize(1);
				
				ray_dir = cam_r.apply(ray_dir);
				
				// if (x == width/2 && y == height/2) {
					// printf("  Emitted from (%.2f, %.2f, %.2f) --> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
				// }
				
				// Advance the ray until it enters the samples.
				// Get the value which, when multiplied by ray_dir, produces the vector in the direction
				// of the ray's travel that ends on the edge of the sample data.
				// The ray will be advanced even if it never intersects the samples,
				// in which case its new position will be completely past the samples,
				// allowing it to be ruled out immediately.
				float adv = -1;
				
				// If the position is left of the samples and moving right...
				if (ray_pos.x < -2 && ray_dir.x > 0) {
					if (ray_pos.y < -1.5 && ray_dir.y > 0) {
						adv = fmax((-1.5 - ray_pos.y) / ray_dir.y, (-2 - ray_pos.x) / ray_dir.x);
					}
					else if (ray_pos.y > -1.5 && ray_pos.y < 1.5) {
						adv = (-2 - ray_pos.x) / ray_dir.x;
					}
					else if  (ray_pos.y > 1.5 && ray_dir.y < 0) {
						adv = fmax((1.5 - ray_pos.y) / ray_dir.y, (-2 - ray_pos.x) / ray_dir.x);
					}
				}
				// If the position is directly above or below the samples...
				else if (ray_pos.x > -2 && ray_pos.x < 1) {
					if (ray_pos.y < -1.5 && ray_dir.y > 0) {
						adv = (-1.5 - ray_pos.y) / ray_dir.y;
					}
					if (ray_pos.y > 1.5 && ray_dir.y < 0) {
						adv = (1.5 - ray_pos.y) / ray_dir.y;
					}
				}
				// If the position is right of the samples and moving left...
				if (ray_pos.x > 1 && ray_dir.x < 0) {
					if (ray_pos.y < -1.5 && ray_dir.y > 0) {
						adv = fmax((-1.5 - ray_pos.y) / ray_dir.y, (1 - ray_pos.x) / ray_dir.x);
					}
					else if (ray_pos.y > -1.5 && ray_pos.y < 1.5) {
						adv = (1 - ray_pos.x) / ray_dir.x;
					}
					else if  (ray_pos.y > 1.5 && ray_dir.y < 0) {
						adv = fmax((1.5 - ray_pos.y) / ray_dir.y, (1 - ray_pos.x) / ray_dir.x);
					}
				}
				
				// If advance is -1, then it can not be advanced.
				// The ray may be inside or already past the samples.
				if (adv != -1) {
					ray_pos = ray_pos + ray_dir * adv;
				}
				
				// if (x == width/2 && y == height/2) {
					// printf("  Advanced to (%.2f, %.2f, %.2f) --> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
				// }
				
				// Convert the ray's position to grid space.
				ray_pos.x = (ray_pos.x + 2.0) / 3 * sample_width;
				ray_pos.y = (ray_pos.y + 1.5) / 3 * sample_height;
				
				ray_dir.x = ray_dir.x / 3 * sample_width;
				ray_dir.y = ray_dir.y / 3 * sample_height;
				
				ray_dir = ray_dir.normalize();
				
				// if (x == width/2 && y == height/2) {
					// printf("  Converted to (%.2f, %.2f, %.2f) --> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
				// }
				
				hit = false;
				for (i = 0; true; i++) {
					// If the ray is out of bounds and moving away from the origin, break and write a black pixel.
					if ( (ray_pos.x < 0 && ray_dir.x < 0) || (ray_pos.x > sample_width && ray_dir.x > 0) || (ray_pos.y < 0 && ray_dir.y < 0) || (ray_pos.y > sample_height && ray_dir.y > 0) || i == ray_max ) {
						img[ind  ] = 0;
						img[ind+1] = 0;
						img[ind+2] = 0;
						break;
					}
					
					// If the ray is in bounds, check for collision with a sample.
					if (ray_pos.x >= 0 && ray_pos.x < sample_width && ray_pos.y >= 0 && ray_pos.y < sample_height) {
						// Get the indices of the data corresponding to this pixel.
						xi = (int) ray_pos.x;
						yi = (int) ray_pos.y;
						
						if (yi >= sample_height / 2) {
							yi = sample_height - yi - 1;
						}
						
						// For each sample in this column, check for collision and write to the image if needed.
						for (j = 1; j <= data[xi][yi][0]; j++) {
							if (d_equ(ray_pos.z, data[xi][yi][j], ray_thresh)) {
								rgb col = rgb(hsv((data[xi][yi][j]+2)/4 * 360, 1, 1));
								
								img[ind  ] = col.r * 255;
								img[ind+1] = col.g * 255;
								img[ind+2] = col.b * 255;
								
								hit = true;
								break;
							}
						}
						
						if (hit) {break;}
					}
					
					// Progress the ray.
					// Calculate the distance the ray will travel before passing an integer x value. Do the same for the y value.
					float x_dis;
					float y_dis;
					if (ray_dir.x > 0) {
						x_dis = (floor(ray_pos.x + 1) - ray_pos.x) / ray_dir.x;
					}
					else {
						x_dis = (ceil(ray_pos.x - 1) - ray_pos.x) / ray_dir.x;
					}
					
					if (ray_dir.y > 0) {
						y_dis = (floor(ray_pos.y + 1) - ray_pos.y) / ray_dir.y;
					}
					else {
						y_dis = (ceil(ray_pos.y - 1) - ray_pos.y) / ray_dir.y;
					}
					
					// Travel the smaller of the two values.
					ray_pos = ray_pos + ray_dir * fmin(x_dis, y_dis);
				}
			}
		}
		
		sprintf(fn, "out/%d.ppm", f);
		int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
		
		FILE* fout = fopen(fn, "wb+");
		fwrite(hdr, 1, hdr_len, fout);
		fwrite(img, 1, width*height*3, fout);
		fclose(fout);
		
		f_end = clock() / CLOCKS_PER_SEC;
		
		printf("Completed in %.2f seconds.\n", f_end-f_start);
	}
	
	delete[] fn;
	
	for (int i = 0; i < sample_width; i++) {
		for (int j = 0; j < sample_height / 2; j++) {
			delete[] data[i][j];
		}
		delete[] data[i];
	}
	delete[] data;
	
	double end = (double) clock() / CLOCKS_PER_SEC;
	
	printf("Complete in %.2fs\n", end-start);
	
	return 0;
}
