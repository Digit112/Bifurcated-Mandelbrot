#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
	int width = 512;
	int height = 512;
	
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
	}
	fclose(fin);
	
	fin = fopen("out.bin", "rb");
	if (fin == NULL) {
		printf("Open failure.\n");
		return 0;
	}
	
	// We don't know how big of an array we'll need. So, we fill this array with values and flush it whenever it fills up.
	uint8_t* img = (uint8_t*) malloc(sizeof(uint8_t) * width * height / 2);
	
	float size_f;
	int size;
	
	for (int i = 0; i < width*(height/2); i++) {
		if (i % 10000 == 0) {
			printf("Sample %d...\n", i);
		}
		
		int read = fread(&size_f, 4, 1, fin);
		if (read != 1) {
			printf("EOF\n");
			break;
		}
		
		size = (int) size_f;
		
		if (size == 0) {
			img[i] = 0;
		}
		else {
			img[i] = (int) (((-1 / (2 * ((float) size / 5 + 1))) + 1) * 255);
		}
		
		read = fseek(fin, size*4, SEEK_CUR);
		
		if (read != 0) {
			printf("Fseek: %d\n", read);
			break;
		}
	}
	
	fclose(fin);
	
	FILE* fout = fopen("out.pgm", "wb");
	if (fout == NULL) {
		printf("Open failure.\n");
		return 0;
	}
	
	char hdr[64];
	int hdr_len = sprintf(hdr, "P5 %d %d 255 ", width, height / 2);
	
	fwrite(hdr, 1, hdr_len, fout);
	fwrite(img, 1, width*(height/2), fout);
	fclose(fout);
	
	return 0;
}