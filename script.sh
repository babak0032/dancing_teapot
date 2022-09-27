#!/bin/sh

N_train="60000"
N_test="5000"
theta="3.141592653589793"
max_theta=$(echo "scale=4; $theta*2/4" | bc)
K="6"
nj="24"

### SO2
group="so2"

## Teapot
objfile="teapot_small.obj"
init_rot="[-0.6283185,0.6283185,0.]"

# constant_velocity
mode="constant_velocity"

python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/teapots_K=${K}_${N_train}_${group}_cv_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/teapots_K=${K}_${N_test}_${group}_cv_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

# small_acceleration
mode="small_acceleration"
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/teapots_K=${K}_${N_train}_${group}_sa_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/teapots_K=${K}_${N_test}_${group}_sa_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

## Airplane
objfile="airplane.obj"
init_rot="[-1.5707963,0.6283185,0.]"

# constant_velocity
mode="constant_velocity"
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/airplanes_K=${K}_${N_train}_${group}_cv_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/airplanes_K=${K}_${N_test}_${group}_cv_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

# small_acceleration
mode="small_acceleration"
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/airplanes_K=${K}_${N_train}_${group}_sa_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/airplanes_K=${K}_${N_test}_${group}_sa_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

### SO3
group="so3"

## Teapot
objfile="teapot_small.obj"
init_rot="[-0.6283185,0.6283185,0.]"

# constant_velocity
mode="constant_velocity"

python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/teapots_K=${K}_${N_train}_${group}_cv_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/teapots_K=${K}_${N_test}_${group}_cv_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

# small_acceleration
mode="small_acceleration"
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/teapots_K=${K}_${N_train}_${group}_sa_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/teapots_K=${K}_${N_test}_${group}_sa_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

## Airplane
objfile="airplane.obj"
init_rot="[-1.5707963,0.6283185,0.]"

# constant_velocity
mode="constant_velocity"
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/airplanes_K=${K}_${N_train}_${group}_cv_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/airplanes_K=${K}_${N_test}_${group}_cv_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile

# small_acceleration
mode="small_acceleration"
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_train --fname="../group-vae/data/airplanes_K=${K}_${N_train}_${group}_sa_aor=2_theta=2pi-4_train.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
python dancing_teapot/teapot_sequence_env_parallel.py --num-timesteps=$N_test --fname="../group-vae/data/airplanes_K=${K}_${N_test}_${group}_sa_aor=2_theta=2pi-4_test.h5" --seed=0 --num-jobs=$nj --K=$K --group=$group --mode=$mode --init-rot=$init_rot --axis-of-rotation="2" --max-step-size=$max_theta --obj-file=$objfile
