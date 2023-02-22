#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "color_GPU.hpp"
#include "vec_GPU.hpp"

bool __device__ d_equ(double a, double b, double epsilon) {
	return abs(a-b) < epsilon;
}

void __global__ render(int sample_width, int sample_height, float* data, int* indexing, uint8_t* img, int width, int height, vecd3 cam_p, quaternion cam_r) {
	float theta = 2 * 3.14159/3;
	int xi; int yi;
	bool hit;
	
	float ray_thresh = 0.003;
	int ray_max = 60000;
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	float theta_h = theta * height / width;
	
	float xt = (float) x/width * theta - theta/2;
	float yt = (float) y/height * theta_h - theta_h/2;

	int ind = ((height - y - 1)*width + x)*3;

	// Create a ray at this camera moving in a direction based on the pixel this ray is coming from.
	vecd3 ray_pos = cam_p;

	vecd3 ray_dir(1, tan(xt), tan(yt));
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
	for (int i = 0; true; i++) {
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
			
			// Locate the appropriate sample.
			int samp_ind = yi*sample_width + xi;
			int num_jumps = samp_ind % 16;
			samp_ind = indexing[samp_ind / 16];
			
			for (int k = 0; k < num_jumps; k++) {
				samp_ind += data[samp_ind] + 1;
			}
			
			// For each sample in this column, check for collision and write to the image if needed.
			for (int j = 1; j <= data[samp_ind]; j++) {
				if (d_equ(ray_pos.z, data[samp_ind + j], ray_thresh)) {
					rgb col = rgb(hsv((data[samp_ind + j]+2)/4 * 360, 1, 1));
					
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
	
	dim3 block(32, 32);
	dim3 grid(width/32, height/32);
	
	cudaError_t err;
	
	// Create the buffer to hold the image to write.
	uint8_t* img;
	err = cudaMallocManaged(&img, width*height*3);
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
	char hdr[64];
	
	vecd3 cam_p_i(-3, 0, 1);
	quaternion cam_r_i(vecd3(0, 1, 0), 20*d2r);
	
	double start = (double) clock() / CLOCKS_PER_SEC;
	
	// Load data
	FILE* fp = fopen("out.bin", "rb");
	if (fp == NULL) {
		printf("Error opening samples.\n");
		return 1;
	}
	
	fseek(fp, 0, SEEK_END);
	
	int fp_n = ftell(fp)/4;
	fseek(fp, 0, SEEK_SET);
	
	printf("Loading...\n");
	float* data;
	err = cudaMallocManaged(&data, fp_n*sizeof(float));
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
	
	int read = fread(data, sizeof(float), fp_n, fp);
	if (read != fp_n) {
		printf("Error reading samples.\n");
		return 1;
	}
	
	fclose(fp);
	
	// Construct indexing table.
	int* indexing;
	err = cudaMallocManaged(&indexing, ceil((double) sample_width * sample_height / 32 * sizeof(int)));
	printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
	
	printf("Constructing the indexing table.\n");
	int size;
	int ind = 0;
	for (int i = 0; i < sample_width * sample_height / 2; i++) {
		if (i % 16 == 0) {
//			printf("%d\n", i/16);
			indexing[i/16] = ind;
		}
		
//		printf("  %d: %d\n", i, ind);
		size = (int) data[ind];
		if (size != data[ind]) {
			printf("Size error.\n");
			return 1;
		}
		ind += size + 1;
	}
	
	float f_start; float f_end;
	float rot_theta;
	
	vecd3 cam_p;
	quaternion cam_r;
	
	vecd3 ray_pos; vecd3 ray_dir;
	
	char* fn = new char[32];
	for (int f = 0; f < frames; f++) {
		printf("Frame %d... \n", f);
		fflush(stdout);
		
		f_start = clock() / CLOCKS_PER_SEC;
		
		rot_theta = 3.14159*2 * f / frames;
		
		quaternion orbit( vecd3(0, 0, 1), rot_theta );
		
		cam_p = orbit.apply(cam_p_i) - vecd3(0.5, 0, 0);
		cam_r = quaternion::hamilton(orbit, cam_r_i);
		
		// Render
		render<<<grid, block>>>(sample_width, sample_height, data, indexing, img, width, height, cam_p, cam_r);
	
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
		
		// Write the rendered image to file.
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
	
	cudaFree(data);
	cudaFree(indexing);
	cudaFree(img);
	
	double end = (double) clock() / CLOCKS_PER_SEC;
	
	printf("Complete in %.2fs\n", end-start);
	
	return 0;
}