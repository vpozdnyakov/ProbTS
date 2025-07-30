export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps

for SEED in 0
do
    for DATASET in 'm4_weekly'
    do
        for MODEL in 'binformer'
        do
            python run.py --config config/m4/${DATASET}/${MODEL}.yaml --seed_everything ${SEED} \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR}
        done
    done
done