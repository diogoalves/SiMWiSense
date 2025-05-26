#!/bin/sh

pushd .

cd ../Python_Code

# Figure 3 - Classification accuracy as function of sensing proximity. (18 models)
python baseline_proximity.py Classroom m1 m1 Classroom_m1_m1.h5 242
python baseline_proximity.py Classroom m1 m2 Classroom_m1_m2.h5 242
python baseline_proximity.py Classroom m1 m3 Classroom_m1_m3.h5 242
python baseline_proximity.py Classroom m2 m1 Classroom_m2_m1.h5 242
python baseline_proximity.py Classroom m2 m2 Classroom_m2_m2.h5 242
python baseline_proximity.py Classroom m2 m3 Classroom_m2_m3.h5 242
python baseline_proximity.py Classroom m3 m1 Classroom_m3_m1.h5 242
python baseline_proximity.py Classroom m3 m2 Classroom_m3_m2.h5 242
python baseline_proximity.py Classroom m3 m3 Classroom_m3_m3.h5 242
python baseline_proximity.py Office m1 m1 Office_m1_m1.h5 242
python baseline_proximity.py Office m1 m2 Office_m1_m2.h5 242
python baseline_proximity.py Office m1 m3 Office_m1_m3.h5 242
python baseline_proximity.py Office m2 m1 Office_m2_m1.h5 242
python baseline_proximity.py Office m2 m2 Office_m2_m2.h5 242
python baseline_proximity.py Office m2 m3 Office_m2_m3.h5 242
python baseline_proximity.py Office m3 m1 Office_m3_m1.h5 242
python baseline_proximity.py Office m3 m2 Office_m3_m2.h5 242
python baseline_proximity.py Office m3 m3 Office_m3_m3.h5 242

# Figure 11: Subject identification in SiMWiSense with baseline CNN (6 models)
python baseline_coarse.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_242.h5 242
python baseline_coarse.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_242.h5 242
python baseline_coarse.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_242.h5 242
python baseline_coarse.py Office m1 Office m1 Office_m1_Office_m1_242.h5 242
python baseline_coarse.py Office m2 Office m2 Office_m2_Office_m2_242.h5 242
python baseline_coarse.py Office m3 Office m3 Office_m3_Office_m3_242.h5 242

# Figure 12: Performance of SiMWiSense at three different environments with baseline CNN. (6 models)
python baseline_fine_grained.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_242.h5 242
python baseline_fine_grained.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_242.h5 242
python baseline_fine_grained.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_242.h5 242
python baseline_fine_grained.py Office m1 Office m1 Office_m1_Office_m1_242.h5 242
python baseline_fine_grained.py Office m2 Office m2 Office_m2_Office_m2_242.h5 242
python baseline_fine_grained.py Office m3 Office m3 Office_m3_Office_m3_242.h5 242

# Figure 13: Performance of SiMWiSense with baseline CNN as a function of number of subcarriers (environment: classroom). (12 models)
python baseline_fine_grained.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_20.h5 20
python baseline_fine_grained.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_40.h5 40
python baseline_fine_grained.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_80.h5 80
python baseline_fine_grained.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_160.h5 160
python baseline_fine_grained.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_20.h5 20
python baseline_fine_grained.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_40.h5 40
python baseline_fine_grained.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_80.h5 80
python baseline_fine_grained.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_160.h5 160
python baseline_fine_grained.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_20.h5 20
python baseline_fine_grained.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_40.h5 40
python baseline_fine_grained.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_80.h5 80
python baseline_fine_grained.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_160.h5 160

# Figure 15: Performance of FREL in simultaneous activity sensing with new untrained environments.
python baseline_fine_grained.py Classroom m1 Office m1 Classroom_m1_Office_m1_242.h5 242
python baseline_fine_grained.py Classroom m2 Office m2 Classroom_m2_Office_m2_242.h5 242
python baseline_fine_grained.py Classroom m3 Office m3 Classroom_m3_Office_m3_242.h5 242
python baseline_fine_grained.py Office m1 Classroom m1 Office_m1_Classroom_m1_242.h5 242
python baseline_fine_grained.py Office m2 Classroom m2 Office_m2_Classroom_m2_242.h5 242
python baseline_fine_grained.py Office m3 Classroom m3 Office_m3_Classroom_m3_242.h5 242
# faltam modelos FREL (mais 6)

# Figure 16: Performance comparison of FREL with FSEL in new untrained environments. (12 models)

# Figure 18: Performance of FREL as the subject identifier in untrained environments. (12 models)

# Figure 19: Performance comparison of FREL and FSEL as the subject identifier in untrained environments (12 models)

# Figure 20: Performance of FREL with new untrained monitors. (12 models).






popd