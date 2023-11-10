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
# make train3 CONFIG=exp033 # 0/5folds
