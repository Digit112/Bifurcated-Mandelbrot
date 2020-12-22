float4  fhamiltonf(float4 a, float4 b) {
	float4 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
	return out;
}

float3  vfhamiltonf(float4 a, float4 b) {
	float3 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
	return out;
}

kernel void render(int width, int height, write_only image2d_t out, global float* data,
		double cpx, double cpy, double cpz, double crx, double cry, double crz, double crw) {
	
	const float theta = 3.14159/3;
	
	const int data_width = 1000;
	const int data_height = 1000;
	
	const float ray_jump = 0.0001;
	const float ray_thresh = 0.002;
	const int ray_max = 50000;
	
	const int2 pix = {get_global_id(0), get_global_id(1)};
	
	float3 ray_pos = {cpx, cpy, cpz};
	float3 ray_dir = {1, tan((float) pix.x / width  * theta - theta/2), tan((float) pix.y / height * theta - theta/2)};
	
	const float m = length(ray_dir) / ray_jump;
	ray_dir.x = ray_dir.x/m;
	ray_dir.y = ray_dir.y/m;
	ray_dir.z = ray_dir.z/m;
	
	const float4 cam_r = {crx, cry, crz, crw};
	const float4 cam_nr = {-crx, -cry, -crz, crw};
	const float4 ray_dir_q = {ray_dir.x, ray_dir.y, ray_dir.z, 0};
	
	const uint4 black = {0, 0, 0, 0};
	uint4 color = {255, 255, 255, 255};
	
	ray_dir = vfhamiltonf(fhamiltonf(cam_r, ray_dir_q), cam_nr);
	
	int xi; int yi;
	int ind; int size;
	int i; int j;
	for (i = 0; 1; i++) {
		if (i == ray_max) {
			write_imageui(out, pix, black);
			return;
		}
		
		if (ray_pos.x > -1.99 && ray_pos.x < 0.99 && ray_pos.y > -1.49 && ray_pos.y < 1.49) {
			xi = (int) ((ray_pos.x + 2.0)/3*data_width);
			yi = (int) ((ray_pos.y + 1.5)/3*data_height);
			
			ind  = data[xi + yi*data_width];
			size = data[ind];
			for (j = ind+1; j <= ind+size; j++) {
				if (fabs(ray_pos.z - data[j]) < ray_thresh) {
					color.x = (data[j]+2)/4*255;
					write_imageui(out, pix, color);
					return;
				}
			}
		}
		
		ray_pos = ray_pos + ray_dir;
	}
}





























