#!/bin/bash

# compare sample_per_epoch 5000 and 2500
# make train2 CONFIG=exp016
# make train2 CONFIG=exp017

# compare base_filters 64, 64*2, 64*3
# make train3 CONFIG=exp023
make train3 CONFIG=exp024
make train3 CONFIG=exp025

# compare time_mask_param=80(023), 80*2(026), 80*3(027), 40(028)
# make train3 CONFIG=exp023
make train3 CONFIG=exp026
make train3 CONFIG=exp027
make train3 CONFIG=exp028
