#!/bin/bash


export XBD_ENV=1
export XBD_HOOK_FOR_EMBEDDING_COLUMN=1
export XBD_CPP_OPS_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0

tf_config=`python -c "import json,os; print json.loads(os.getenv('TF_CONFIG')).get('cluster').get('ps')"`
if [ "$tf_config" = "None" ] ; then
unset TF_CONFIG
fi

cd super_topic/dssm
python xbdProcessor.py \
--model_id=d_m_17773 \
--user=chengsi2 \
--sample_path=hdfs://ns-fed/wbml/kafka2hdfs/super-topic-ulike-realread-realtime-sample-flat \
--is_hour_partition='' \
--dt_partition=20230617 \
--user_id=uid \
--item_id=r_stid \
--active_fn=relu,relu,tanh \
--loss=log_loss \
--layer_nodes=256,128,64 \
--dropout_rate=0.1 \
--batch_size=1024 \
--l2_reg=0.001 \
--label=is_addatten \
--embed_dim=64 \
--learning_rate=0.001 \
--mode=train \
--needOffline='' \
