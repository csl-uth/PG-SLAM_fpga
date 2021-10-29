# Fastest Approximate Configuration with untracked frames (C1)

Configuration C1 uses for each kernel the following optimizations:

* Bilateral Filter (HW) :  BF_Pipe, BF_Unroll, BF_Off, BF_Pad, BF_Coeff, BF_Range
* Integration (HW)      :  Int_Pipe, Int_Unroll, Int_Inter, Int_NCU, Int_SLP, Int_HP, Int_Br, Int_FPOp
* FuseVoxelGrids (HW)   :  F_PGKf, F_LP, F_HP, F_FPOp, F_Cnst, F_SkIn, F_AllInOne (invocation)
* Tracking (SW)			    :  Tr_LP
* Raycast (SW)          :  R_Step, R_LP, R_TrInt, R_Fast, R_Rate
* Parameters            :  kfr=150, fts=150, pgo_m=1, icp=0.01, pd0-pd2=(2,0,0)
