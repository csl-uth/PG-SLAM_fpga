#define _HW_


#include <kfusion.h>
#include <string.h>


extern "C" {
#ifdef FUSE_ALLINONE_5
	#ifdef FUSE_HP
	#ifdef FUSE_UNROLL_APPROX
	void fuseVolumesKernel( short2 *dstVol_r1, short2 *dstVol_r2,
							short2 *dstVol_w1, short2 *dstVol_w2,
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
		#pragma HLS INTERFACE m_axi  port=dstVol_r1 offset=slave bundle=dstVol_r1
		#pragma HLS INTERFACE s_axilite port=dstVol_r1 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_r2 offset=slave bundle=dstVol_r2
		#pragma HLS INTERFACE s_axilite port=dstVol_r2 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_w1 offset=slave bundle=dstVol_w1
		#pragma HLS INTERFACE s_axilite port=dstVol_w1 bundle=control

		#pragma HLS INTERFACE m_axi  port=dstVol_w2 offset=slave bundle=dstVol_w2
		#pragma HLS INTERFACE s_axilite port=dstVol_w2 bundle=control

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

		#pragma HLS DATA_PACK variable=dstVol_r1
		#pragma HLS DATA_PACK variable=dstVol_r2
		#pragma HLS DATA_PACK variable=dstVol_w1
		#pragma HLS DATA_PACK variable=dstVol_w2

		#pragma HLS DATA_PACK variable=size


		uint3 vres;
		vres.x = size->x;
		vres.y = size->y;
		vres.z = size->z;

		unsigned int z, y, x;

		for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256/FUSE_NCU; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256/FUSE_NCU; x+=2) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;

					// buffer #0
					floatH tsdf = tsdf_buffer[index];

					floatH w_interp = 1;
					
					#ifdef FUSE_FPOP
						floatH2 p_data_1 = make_floatH2(dstVol_r1[index].x>>15, dstVol_r1[index].y);
					#else
						floatH2 p_data_1 = make_floatH2(dstVol_r1[index].x*0.00003051944088f,dstVol_r1[index].y);
					#endif
					#ifdef FUSE_FPOP
						floatH2 p_data_2 = make_floatH2(dstVol_r2[index+1].x>>15, dstVol_r2[index+1].y);
					#else
						floatH2 p_data_2 = make_floatH2(dstVol_r2[index+1].x*0.00003051944088f,dstVol_r2[index+1].y);
					#endif
					
					if(tsdf != 1.0){
						floatH new_w = p_data_1.y + w_interp;

						p_data_1.x = clamp( (p_data_1.y*p_data_1.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_1.y = fminf(new_w, maxweight);

						floatH new_w2 = p_data_2.y + w_interp;

						p_data_2.x = clamp( (p_data_2.y*p_data_2.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_2.y = fminf(new_w, maxweight);
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						floatH new_w = p_data_1.y + w_interp;

						p_data_1.x = clamp( (p_data_1.y*p_data_1.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_1.y = fminf(new_w, maxweight);

						floatH new_w2 = p_data_2.y + w_interp;

						p_data_2.x = clamp( (p_data_2.y*p_data_2.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_2.y = fminf(new_w, maxweight);
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						floatH new_w = p_data_1.y + w_interp;

						p_data_1.x = clamp( (p_data_1.y*p_data_1.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_1.y = fminf(new_w, maxweight);

						floatH new_w2 = p_data_2.y + w_interp;

						p_data_2.x = clamp( (p_data_2.y*p_data_2.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_2.y = fminf(new_w, maxweight);
					}

					// buffer #3
					tsdf = tsdf_buffer_3[index];
					if(tsdf != 1.0){
						floatH new_w = p_data_1.y + w_interp;

						p_data_1.x = clamp( (p_data_1.y*p_data_1.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_1.y = fminf(new_w, maxweight);

						floatH new_w2 = p_data_2.y + w_interp;

						p_data_2.x = clamp( (p_data_2.y*p_data_2.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_2.y = fminf(new_w, maxweight);
					}
					
					// buffer #4
					tsdf = tsdf_buffer_4[index];
					if(tsdf != 1.0){
						floatH new_w = p_data_1.y + w_interp;

						p_data_1.x = clamp( (p_data_1.y*p_data_1.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_1.y = fminf(new_w, maxweight);

						floatH new_w2 = p_data_2.y + w_interp;

						p_data_2.x = clamp( (p_data_2.y*p_data_2.x + w_interp*tsdf) / new_w, -1.f, 1.f);
						p_data_2.y = fminf(new_w, maxweight);
					}

					#ifdef FUSE_FPOP
						dstVol_w1[index] = make_short2(int(p_data_1.x)<<15, p_data_1.y);
					#else
						dstVol_w1[index] = make_short2(p_data_1.x*32766.0f, p_data_1.y);
					#endif

					#ifdef FUSE_FPOP
						dstVol_w2[index+1] = make_short2(int(p_data_2.x)<<15, p_data_2.y);
					#else
						dstVol_w2[index+1] = make_short2(p_data_2.x*32766.0f, p_data_2.y);
					#endif
				}
			}
		}
		
	}
	#else // FUSE_UNROLL_APPROX
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

		for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256/FUSE_NCU; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256/FUSE_NCU; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;
					
					// buffer #0
					floatH tsdf = tsdf_buffer[index];

					floatH w_interp = 1;
					
					#ifdef FUSE_FPOP
						floatH2 p_data = make_floatH2(dstVol_1[index].x>>15, dstVol_1[index].y);
					#else
						floatH2 p_data = make_floatH2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					#endif
					
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #3
					tsdf = tsdf_buffer_3[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}
					
					// buffer #4
					tsdf = tsdf_buffer_4[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					#ifdef FUSE_FPOP
						dstVol_2[index] = make_short2(int(p_data.x)<<15, p_data.y);
					#else
						dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					#endif
				}
			}
		}
		
	}
	#endif // FUSE_UNROLL_APPROX
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

		for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256/FUSE_NCU; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256/FUSE_NCU; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;

					// buffer #0
					float tsdf = tsdf_buffer[index];

					float w_interp = 1;
					
					
					#ifdef FUSE_FPOP
						float2 p_data = make_float2(dstVol_1[index].x>>15, dstVol_1[index].y);
					#else
						float2 p_data = make_float2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					#endif
					
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #3
					tsdf = tsdf_buffer_3[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}
					
					// buffer #4
					tsdf = tsdf_buffer_4[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					#ifdef FUSE_FPOP
						dstVol_2[index] = make_short2(int(p_data.x)<<15, p_data.y);
					#else
						dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					#endif
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

		for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256/FUSE_NCU; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256/FUSE_NCU; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;
					
					// buffer #0
					floatH tsdf = tsdf_buffer[index];

					floatH w_interp = 1;
					
					#ifdef FUSE_FPOP
						floatH2 p_data = make_floatH2(dstVol_1[index].x>>15, dstVol_1[index].y);
					#else
						floatH2 p_data = make_floatH2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					#endif
					
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							floatH new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					#ifdef FUSE_FPOP
						dstVol_2[index] = make_short2(int(p_data.x)<<15, p_data.y);
					#else
						dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					#endif
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

		for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
			int z_idx = z*vres.x*vres.y;
			for (y = 0; y < 256/FUSE_NCU; y++) {
				int idx = y*vres.x + z_idx;
				X_LOOP: for (x = 0; x < 256/FUSE_NCU; x++) {
					#pragma HLS PIPELINE II=1
					uint index = x + idx;

					// buffer #0
					float tsdf = tsdf_buffer[index];

					float w_interp = 1;
					
					
					#ifdef FUSE_FPOP
						float2 p_data = make_float2(dstVol_1[index].x>>15, dstVol_1[index].y);
					#else
						float2 p_data = make_float2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
					#endif
					
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #1
					tsdf = tsdf_buffer_1[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					// buffer #2
					tsdf = tsdf_buffer_2[index];
					if(tsdf != 1.0){
						#ifdef FUSE_ELIMCONST
							p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
							p_data.y = fminf(p_data.y, maxweight);
						#else
							float new_w = p_data.y + w_interp;

							p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
							p_data.y = fminf(new_w, maxweight);
						#endif
					}

					#ifdef FUSE_FPOP
						dstVol_2[index] = make_short2(int(p_data.x)<<15, p_data.y);
					#else
						dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
					#endif
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

	for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
		int z_idx = z*vres.x*vres.y;
		for (y = 0; y < 256/FUSE_NCU; y++) {
			int idx = y*vres.x + z_idx;
			X_LOOP: for (x = 0; x < 256/FUSE_NCU; x++) {
				#pragma HLS PIPELINE II=1

				uint index = x + idx;
				
				floatH tsdf = tsdf_buffer[index];

				if(tsdf == 1.0)
					continue;

				floatH w_interp = 1;
				
				#ifdef FUSE_FPOP
					floatH2 p_data = make_floatH2(dstVol_1[index].x>>15, dstVol_1[index].y);
				#else
					floatH2 p_data = make_floatH2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
				#endif
				
				#ifdef FUSE_ELIMCONST
					p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
					p_data.y = fminf(p_data.y, maxweight);
				#else
					floatH new_w = p_data.y + w_interp;
				
					p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
					p_data.y = fminf(new_w, maxweight);
				#endif                
				
				#ifdef FUSE_FPOP
					dstVol_2[index] = make_short2(int(p_data.x)<<15, p_data.y);
				#else
					dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
				#endif
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

	for (z = z_offset; z < 256/FUSE_NCU; z+=lp_step) {
		int z_idx = z*vres.x*vres.y;
		for (y = 0; y < 256/FUSE_NCU; y++) {
			int idx = y*vres.x + z_idx;
			X_LOOP: for (x = 0; x < 256/FUSE_NCU; x++) {
				#pragma HLS PIPELINE II=1

				uint index = x + idx;
				
				float tsdf = tsdf_buffer[index];

				if(tsdf == 1.0)
					continue;

				float w_interp = 1;
				
				#ifdef FUSE_FPOP
					float2 p_data = make_float2(dstVol_1[index].x>>15, dstVol_1[index].y);
				#else
					float2 p_data = make_float2(dstVol_1[index].x*0.00003051944088f,dstVol_1[index].y);
				#endif
				
				#ifdef FUSE_ELIMCONST
					p_data.x = clamp(p_data.x + (tsdf / p_data.y), -1.f, 1.f);
					p_data.y = fminf(p_data.y, maxweight);
				#else
					float new_w = p_data.y + w_interp;
				
					p_data.x = clamp( (p_data.y*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
					p_data.y = fminf(new_w, maxweight);  
				#endif              
				
				#ifdef FUSE_FPOP
					dstVol_2[index] = make_short2(int(p_data.x)<<15, p_data.y);
				#else
					dstVol_2[index] = make_short2(p_data.x*32766.0f, p_data.y);
				#endif
			}
		}
	}
	
}
#endif // FUSE_HP
#endif // FUSE_ALLINONE_3
#endif // FUSE_ALLINONE_5
}
