#!/bin/bash

pushd .

cd ../Python_Code

python baseline_fine_grained.py Classroom m1 Classroom m1 Classroom_m1_Classroom_m1_242.h5 242
python baseline_fine_grained.py Classroom m2 Classroom m2 Classroom_m2_Classroom_m2_242.h5 242
python baseline_fine_grained.py Classroom m3 Classroom m3 Classroom_m3_Classroom_m3_242.h5 242
python baseline_fine_grained.py Office m1 Office m1 Office_m1_Office_m1_242.h5 242
python baseline_fine_grained.py Office m2 Office m2 Office_m2_Office_m2_242.h5 242
python baseline_fine_grained.py Office m3 Office m3 Office_m3_Office_m3_242.h5 242

popd