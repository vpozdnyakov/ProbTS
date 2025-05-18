export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps

for SEED in 1 2 3 4 5
do
    for DATASET in 'm4_daily' 'm4_weekly' 'tourism_monthly'
    do
        for MODEL in 'dlinear' 'gru_nvp' 'patchtst' 'timegrad'
        do
            python run.py --config config/m4/${DATASET}/${MODEL}.yaml --seed_everything ${SEED} \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR}
        done
    done
done