#define _HW_


#include <kfusion.h>
#include <string.h>


extern "C" {
#ifdef FUSE_ALLINONE_5
	#ifdef FUSE_HP
	void fuseVolumesKernel( short2 *dstVol_1, short2 *dstVol_2,
							floatH *tsdf_buffer,
							floatH *tsdf_buffer_1,
							floatH *tsdf_buffer_2,
							floatH *tsdf_buffer_3,
							floatH *tsdf_buffer_4,
							floatH maxweight,
							uint4 *size,	// one element (float3)
							int lp_step,
							int z_offset)
	{
		#pragma HLS INTERFACE s_axilite port=return bundle=control
		#pragma HLS INTERFACE m_axi  port=dstVol_1 offset=slave bundle=dstVol_1
		#pragma HLS INTERFACE s_axilite port=dstVol_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_2 offset=slave bundle=dstVol_2
		#pragma HLS INTERFACE s_axilite port=dstVol_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer offset=slave bundle=tsdf_buffer
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_1 offset=slave bundle=tsdf_buffer_1
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_2 offset=slave bundle=tsdf_buffer_2
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_3 offset=slave bundle=tsdf_buffer_3
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_3 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_4 offset=slave bundle=tsdf_buffer_4
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_4 bundle=control

		#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
		#pragma HLS INTERFACE s_axilite port=size bundle=control

		#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
		#pragma HLS INTERFACE s_axilite port=lp_step bundle=control
		#pragma HLS INTERFACE s_axilite port=z_offset bundle=control

		#pragma HLS DATA_PACK variable=dstVol_1
		#pragma HLS DATA_PACK variable=dstVol_2
		#pragma HLS DATA_PACK variable=size


		uint3 vres;
		vres.x = size->x;
		vres.y = size->y;
		vres.z = size->z;

		unsigned int z, y, x;

		for (z = z_offset; z < 256; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;
					
					// buffer #0
					floatH tsdf = tsdf_buffer[index];

					floatH w_interp = 1;
					
					floatH2 p_data = make_floatH2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #3
					tsdf = tsdf_buffer_3[index];
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}
					
					// buffer #4
					tsdf = tsdf_buffer_4[index];
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}
					dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					
				}
			}
		}
		
	}
	#else 
	void fuseVolumesKernel( short2 *dstVol_1, short2 *dstVol_2,
							float *tsdf_buffer,
							float *tsdf_buffer_1,
							float *tsdf_buffer_2,
							float *tsdf_buffer_3,
							float *tsdf_buffer_4,
							float maxweight,
							uint4 *size,	// one element (float3)
							int lp_step,
							int z_offset)
	{
		#pragma HLS INTERFACE s_axilite port=return bundle=control
		#pragma HLS INTERFACE m_axi  port=dstVol_1 offset=slave bundle=dstVol_1
		#pragma HLS INTERFACE s_axilite port=dstVol_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_2 offset=slave bundle=dstVol_2
		#pragma HLS INTERFACE s_axilite port=dstVol_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer offset=slave bundle=tsdf_buffer
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_1 offset=slave bundle=tsdf_buffer_1
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_2 offset=slave bundle=tsdf_buffer_2
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_3 offset=slave bundle=tsdf_buffer_3
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_3 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_4 offset=slave bundle=tsdf_buffer_4
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_4 bundle=control

		#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
		#pragma HLS INTERFACE s_axilite port=size bundle=control

		#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
		#pragma HLS INTERFACE s_axilite port=lp_step bundle=control
		#pragma HLS INTERFACE s_axilite port=z_offset bundle=control

		#pragma HLS DATA_PACK variable=dstVol_1
		#pragma HLS DATA_PACK variable=dstVol_2
		#pragma HLS DATA_PACK variable=size


		uint3 vres;
		vres.x = size->x;
		vres.y = size->y;
		vres.z = size->z;

		unsigned int z, y, x;

		for (z = z_offset; z < 256; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;

					float tsdf = tsdf_buffer[index];

					float w_interp = 1;
					
					
					float2 p_data = make_float2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #3
					tsdf = tsdf_buffer_3[index];
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}
					
					// buffer #4
					tsdf = tsdf_buffer_4[index];
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					
				}
			}
		}
		
	}
	#endif // FUSE_HP
#else // NOT FUSE_ALLINONE_5

#ifdef FUSE_ALLINONE_3
	#ifdef FUSE_HP
	void fuseVolumesKernel( short2 *dstVol_1, short2 *dstVol_2,
							floatH *tsdf_buffer,
							floatH *tsdf_buffer_1,
							floatH *tsdf_buffer_2,
							floatH maxweight,
							uint4 *size,	// one element (float3)
							int lp_step,
							int z_offset)
	{
		#pragma HLS INTERFACE s_axilite port=return bundle=control
		#pragma HLS INTERFACE m_axi  port=dstVol_1 offset=slave bundle=dstVol_1
		#pragma HLS INTERFACE s_axilite port=dstVol_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_2 offset=slave bundle=dstVol_2
		#pragma HLS INTERFACE s_axilite port=dstVol_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer offset=slave bundle=tsdf_buffer
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_1 offset=slave bundle=tsdf_buffer_1
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_2 offset=slave bundle=tsdf_buffer_2
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
		#pragma HLS INTERFACE s_axilite port=size bundle=control

		#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
		#pragma HLS INTERFACE s_axilite port=lp_step bundle=control
		#pragma HLS INTERFACE s_axilite port=z_offset bundle=control

		#pragma HLS DATA_PACK variable=dstVol_1
		#pragma HLS DATA_PACK variable=dstVol_2
		#pragma HLS DATA_PACK variable=size


		uint3 vres;
		vres.x = size->x;
		vres.y = size->y;
		vres.z = size->z;

		unsigned int z, y, x;

		for (z = z_offset; z < 256; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;
					
					floatH tsdf = tsdf_buffer[index];

					floatH w_interp = 1;
					
					floatH2 p_data = make_floatH2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						floatH new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					
				}
			}
		}
		
	}
	#else 
	void fuseVolumesKernel( short2 *dstVol_1, short2 *dstVol_2,
							float *tsdf_buffer,
							float *tsdf_buffer_1,
							float *tsdf_buffer_2,
							float maxweight,
							uint4 *size,	// one element (float3)
							int lp_step,
							int z_offset)
	{
		#pragma HLS INTERFACE s_axilite port=return bundle=control
		#pragma HLS INTERFACE m_axi  port=dstVol_1 offset=slave bundle=dstVol_1
		#pragma HLS INTERFACE s_axilite port=dstVol_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_2 offset=slave bundle=dstVol_2
		#pragma HLS INTERFACE s_axilite port=dstVol_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer offset=slave bundle=tsdf_buffer
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_1 offset=slave bundle=tsdf_buffer_1
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_1 bundle=control

		#pragma HLS INTERFACE m_axi  port=tsdf_buffer_2 offset=slave bundle=tsdf_buffer_2
		#pragma HLS INTERFACE s_axilite port=tsdf_buffer_2 bundle=control

		#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
		#pragma HLS INTERFACE s_axilite port=size bundle=control

		#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
		#pragma HLS INTERFACE s_axilite port=lp_step bundle=control
		#pragma HLS INTERFACE s_axilite port=z_offset bundle=control

		#pragma HLS DATA_PACK variable=dstVol_1
		#pragma HLS DATA_PACK variable=dstVol_2
		#pragma HLS DATA_PACK variable=size


		uint3 vres;
		vres.x = size->x;
		vres.y = size->y;
		vres.z = size->z;

		unsigned int z, y, x;

		for (z = z_offset; z < 256; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;

					float tsdf = tsdf_buffer[index];

					float w_interp = 1;
					
					
					float2 p_data = make_float2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						float new_w = p_data.y + w_interp;

						p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data.y = fminf(new_w, maxweight);
					}

					dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					
				}
			}
		}
		
	}
	#endif // FUSE_HP
#else // NOT FUSE_ALLINONE_3

#ifdef FUSE_HP
void fuseVolumesKernel( short2 *dstVol_1, short2 *dstVol_2,
						floatH *tsdf_buffer,
						floatH maxweight,
						uint4 *size,	// one element (float3)
						int lp_step,
						int z_offset)
{
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE m_axi  port=dstVol_1 offset=slave bundle=dstVol_1
	#pragma HLS INTERFACE s_axilite port=dstVol_1 bundle=control

	#pragma HLS INTERFACE m_axi  port=dstVol_2 offset=slave bundle=dstVol_2
	#pragma HLS INTERFACE s_axilite port=dstVol_2 bundle=control

	#pragma HLS INTERFACE m_axi  port=tsdf_buffer offset=slave bundle=tsdf_buffer
	#pragma HLS INTERFACE s_axilite port=tsdf_buffer bundle=control

	#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
	#pragma HLS INTERFACE s_axilite port=size bundle=control

	#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
	#pragma HLS INTERFACE s_axilite port=lp_step bundle=control
	#pragma HLS INTERFACE s_axilite port=z_offset bundle=control

	#pragma HLS DATA_PACK variable=dstVol_1
	#pragma HLS DATA_PACK variable=dstVol_2
	#pragma HLS DATA_PACK variable=size


	uint3 vres;
	vres.x = size->x;
	vres.y = size->y;
	vres.z = size->z;

	unsigned int z, y, x;

	for (z = z_offset; z < 256; z+=lp_step) {
		int z_idx = z*vres.x*vres.y;
		for (y = 0; y < 256; y++) {
			int idx = y*vres.x + z_idx;
			X_LOOP: for (x = 0; x < 256; x++) {
				#pragma HLS PIPELINE II=1

				uint index = x + idx;
				
				floatH tsdf = tsdf_buffer[index];

				if(tsdf == 1.0)
					continue;

				floatH w_interp = 1;
				
				floatH2 p_data = make_floatH2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
				
				floatH new_w = p_data.y + w_interp;
				
				p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
				p_data.y = fminf(new_w, maxweight);                
				
				dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
				
			}
		}
	}
	
}
#else
void fuseVolumesKernel( short2 *dstVol_1, short2 *dstVol_2,
						float *tsdf_buffer,
						float maxweight,
						uint4 *size,	// one element (float3)
						int lp_step,
						int z_offset)
{
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE m_axi  port=dstVol_1 offset=slave bundle=dstVol_1
	#pragma HLS INTERFACE s_axilite port=dstVol_1 bundle=control

	#pragma HLS INTERFACE m_axi  port=dstVol_2 offset=slave bundle=dstVol_2
	#pragma HLS INTERFACE s_axilite port=dstVol_2 bundle=control

	#pragma HLS INTERFACE m_axi  port=tsdf_buffer offset=slave bundle=tsdf_buffer
	#pragma HLS INTERFACE s_axilite port=tsdf_buffer bundle=control

	#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
	#pragma HLS INTERFACE s_axilite port=size bundle=control

	#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
	#pragma HLS INTERFACE s_axilite port=lp_step bundle=control
	#pragma HLS INTERFACE s_axilite port=z_offset bundle=control

	#pragma HLS DATA_PACK variable=dstVol_1
	#pragma HLS DATA_PACK variable=dstVol_2
	#pragma HLS DATA_PACK variable=size


	uint3 vres;
	vres.x = size->x;
	vres.y = size->y;
	vres.z = size->z;

	unsigned int z, y, x;

	for (z = z_offset; z < 256; z+=lp_step) {
		int z_idx = z*vres.x*vres.y;
		for (y = 0; y < 256; y++) {
			int idx = y*vres.x + z_idx;
			X_LOOP: for (x = 0; x < 256; x++) {
				#pragma HLS PIPELINE II=1

				uint index = x + idx;
				
				float tsdf = tsdf_buffer[index];

				if(tsdf == 1.0)
					continue;

				float w_interp = 1;
				
				float2 p_data = make_float2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
				float new_w = p_data.y + w_interp;
				
				p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
				p_data.y = fminf(new_w, maxweight);                
				
				dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
				
			}
		}
	}
	
}
#endif // FUSE_HP
#endif // FUSE_ALLINONE_3
#endif // FUSE_ALLINONE_5
}
