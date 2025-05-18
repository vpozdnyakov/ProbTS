export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps

for DATASET in 'm5'
do
    for MODEL in 'gru_nvp' 'patchtst' 'timegrad'
    do
        python run.py --config config/m4/${DATASET}/${MODEL}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR}
    done
done
