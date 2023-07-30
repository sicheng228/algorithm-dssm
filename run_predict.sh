#!/bin/bash

cd super_topic/dssm
export XBD_ENV=1
export XBD_HOOK_FOR_EMBEDDING_COLUMN=1
export XBD_CPP_OPS_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0

tf_config=`python -c "import json,os; print json.loads(os.getenv('TF_CONFIG')).get('cluster').get('ps')"`
if [ "$tf_config" = "None" ] ; then
unset TF_CONFIG
fi

echo "$0, $1, $2"

insID="$1"

echo "insID:${insID}"

#
python xbdProcessor.py \
--model_id=d_m_17773 \
--user=chengsi2 \
--sample_path=hdfs://ns-fed/wbml/wb_oprd_supertopic_algo/warehouse/wbml_ods/st_rec_dssm_feed_stid_features \
--item_embedding_path=hdfs://ns-fed/wbml/wb_oprd_supertopic_algo/warehouse/wbml_dwd/super_topic_prerank_item_embedding_base \
--is_hour_partition=False \
--dt_partition=20230702 \
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
--mode=predict \
--needOffline=True \
--instance_id=${insID} \
