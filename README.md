# Fastest Approximate Configuration without untracked frames (C2)

Configuration C2 uses for each kernel the following optimizations:

* Integration (HW)      :  Int_Pipe, Int_Unroll, Int_Inter, Int_NCU, Int_SLP, Int_HP, Int_Br, Int_FPOp
* FuseVoxelGrids (HW)   :  F_PGKf, F_LP, F_HP, F_FPOp, F_SkIn, F_AllInOne (invocation)
* Bilateral Filter(SW)  :  BF_Off
* Tracking (SW)			    :  Tr_LP
* Raycast (SW)          :  R_Step, R_LP, R_TrInt, R_Fast, R_Rate
* Other approximations  :  O_PGC
* Parameters            :  kfr=100, icp=0.01, pd0-pd2=(3,0,0)
