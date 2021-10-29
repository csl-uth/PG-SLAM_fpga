# Fastest Precise Configuration

The configuration uses all the precise optimizations:

* Bilateral Filter (HW) :  BF_Pipe, BF_Unroll, BF_Pad
* Integration (HW)      :  Int_Pipe, Int_Unroll, Int_Inter
* FuseVoxelGrids (HW)   :  F_Pipe, F_PGKf
* Tracking (SW)			    :  baseline unoptimized implementation
* Raycast (SW)          :  baseline unoptimized implementation
