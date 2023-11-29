#!/bin/bash

# compare sample_per_epoch 5000 and 2500
# make train2 CONFIG=exp016
# make train2 CONFIG=exp017

# make train3 CONFIG=exp021 # 0/10folds 0.7979384087611283
# make train3 CONFIG=exp022 # 0/10folds 0.7914106009566322

# compare base_filters 64, 64*2, 64*3
# make train3 CONFIG=exp023
# make train3 CONFIG=exp024 # 0/10folds 0.7651865235152009
# make train3 CONFIG=exp025 # 0/10folds

# compare time_mask_param=80(023), 80*2(026), 80*3(027), 40(028)
# make train3 CONFIG=exp023 # 0/10folds 0.7624216508138792
# make train3 CONFIG=exp026 # 0/10folds 0.7600345664497541
# make train3 CONFIG=exp027 # 0/10folds 0.6956450697528757
# make train3 CONFIG=exp028 # 0/10folds 0.7430776587806438

# 021+seq_len=24*60*4
# make train3 CONFIG=exp029 # 0/10folds 0.7920901528576572

# 5folds
# seq_len=24*60*4(030), 24*60*6(031), 24*60*8(032)
# make train3 CONFIG=exp030 # 0/5folds 0.692147264363885
# make train3 CONFIG=exp031 # 0/5folds 0.6940936007257762
# make train3 CONFIG=exp032 # 0/5folds 0.7091765153636308

# mixup turn of(032), turn off(033)
# make train3 CONFIG=exp032 # 0/5folds 0.7091765153636308
# make train3 CONFIG=exp033 # 0/5folds 0.7425812436177731

# base => exp033; +encoder_name=mit_b0(034,bs=8*4), mit_b3(035,bs=8*1)
# make train3 CONFIG=exp034 # 0/5folds 0.7652777987441026
# make train3 CONFIG=exp035 # 0/5folds 0.78415255226564

# 35+FocalLoss
# make train3 CONFIG=exp036 # 0/5folds

# compare base_filters 64(037_1), 64*2(038), 64*3(039)
# make train3 CONFIG=exp037_1 # 0/5folds
# make train3 CONFIG=exp038   # 0/5folds 0.7592679604904783
# make train3 CONFIG=exp039   # 0/5folds 0.76563664886476

# compare mixup on raw signal, base 37_1
# make train3 CONFIG=exp040 # 0/5folds

# bg_sampling_rate=0.5(041), 0.4(043), 0.3(044), 0.2(045)
# make train3 CONFIG=exp043 # 0/5folds
# make train3 CONFIG=exp044 # 0/5folds
# make train3 CONFIG=exp045 # 0/5folds

# 41+seq_len=24*60*10(046), 32*16*30(047), 24*60*20(048)
# make train3 CONFIG=exp046 # 0/5folds
# make train3 CONFIG=exp047 # 0/5folds
# make train3 CONFIG=exp048 # 0/5folds

# min_max_norm on anglez
# make train3 CONFIG=exp052 FOLD=0
# make train3 CONFIG=exp052 FOLD=1
# make train3 CONFIG=exp052_1 FOLD=0
# make train3 CONFIG=exp052_1 FOLD=1

# exp067: 063+use_corrected_events=True, 1.6h * 3 = 4.8h
# make train3 CONFIG=exp067 FOLD=1
# make train3 CONFIG=exp067 FOLD=2
# make train3 CONFIG=exp067 FOLD=3

# exp068: 064+use_corrected_events=False, 6.0h * 3 = 18.0h
# make train3 CONFIG=exp068 FOLD=0
# make train3 CONFIG=exp068 FOLD=1
# make train3 CONFIG=exp068 FOLD=2

# exp073,074,75,76
# make train3 CONFIG=exp073 FOLD=0
# make train3 CONFIG=exp073 FOLD=1
# make train3 CONFIG=exp073 FOLD=2

# make train3 CONFIG=exp074 FOLD=0
# make train3 CONFIG=exp074 FOLD=1
# make train3 CONFIG=exp074 FOLD=2

# make train3 CONFIG=exp075 FOLD=0
# make train3 CONFIG=exp075 FOLD=1

# make train3 CONFIG=exp076 FOLD=0
# make train3 CONFIG=exp076 FOLD=1
# make train3 CONFIG=exp076 FOLD=2

make train3 CONFIG=exp078 FOLD=0
make train3 CONFIG=exp078 FOLD=1
make train3 CONFIG=exp078 FOLD=2

# make final sub
rye run python ./scripts/full_train_v3.py --config exp075 --fold 0
# rye run python ./scripts/full_train_v3.py --config exp074 --fold 0
rye run python ./scripts/full_train_v3.py --config exp070 --fold 0
# rye run python ./scripts/full_train_v3.py --config exp067 --fold 0
rye run python ./scripts/full_train_v3.py --config exp064 --fold 0
