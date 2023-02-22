#include <stdio.h>
#include <stdlib.h>

int main() {
	FILE* fin = fopen("out.bin", "rb");
	if (fin == NULL) {
		printf("Open failure.\n");
		return 0;
	}
	
	int* size_dis = (int*) malloc(sizeof(int) * 1600);
	for (int i = 0; i < 1600; i++) {
		size_dis[i] = 0;
	}
	
	float size_f;
	int size;
	
	int sum = 0;
	float avg;
	int num_samples = 0;
	int max_size = 0;
	int num_len_0 = 0;
	
	while (1) {
		if (num_samples % 1000 == 0) {
			printf("Sample %d...\n", num_samples);
		}
		
		int read = fread(&size_f, 4, 1, fin);
		if (read != 1) {
			printf("EOF\n");
			break;
		}
		
		size = (int) size_f;
		
		if (size == 0) {
			num_len_0++;
		}
		size_dis[size/10]++;
		
		read = fseek(fin, size*4, SEEK_CUR);
		
		if (read != 0) {
			printf("Fseek: %d\n", read);
			break;
		}
		
		if (size > max_size) {
			max_size = size;
		}
		
		sum += size;
		num_samples++;
	}
	
	avg = (float) sum / num_samples;
	
	printf("%d, %d, %.2f\n", num_samples, max_size, avg);
	
	printf("0: %d\n", num_len_0);
	
	// int run_sum = 0;
	// for (int i = 0; i < 650; i++) {
		// run_sum += size_dis[i];
		// printf("<%d: %d - %d\n", (i+1)*10, size_dis[i], run_sum);
	// }
	
	fclose(fin);
	
	free(size_dis);
	
	return 0;
}