# coding=utf-8

import xembedding as xbd
from entry_kafka_xbd_dssm import xRuntime
import dataschema
from xembedding.source.BaseSource import *
import json, os
from utils.helios_api import *


class XBDProcess():
    def __init__(self, args):
        self.params = {}

        print('--------------------------------------------------打印args接收到的参数----------------------------------------')
        print(args)

        # 1、get model address
        self.params['instance_id'] = args.get('instance_id')
        self.params['model_name'] = args.get('model_id')
        self.params['mode'] = args.get('mode')
        self.params['train'] = args.get('train', True)
        self.params['model_deploy'] = False
        self.params['model_version'] = args.get('model_version', '001')
        self.params['base_dir'] = 'hdfs://ns-fed/wbml/wb_oprd_supertopic_algo/model/zoo/model_id=' + self.params['model_name']
        self.params['model_root_dir'] = self.params['base_dir']
        self.params['model_dir'] = self.params['base_dir'] + '/checkpoint'
        self.params['model_export_dir'] = self.params['base_dir'] + '/version'
        self.params['model_profile_dir'] = self.params['base_dir'] + '/profile'
        self.params['runtime_stream_eval_dir'] = self.params['base_dir'] + '/eval'
        self.params['runtime_stream_eval'] = bool(args.get('runtime_stream_eval', True))
        self.params['train_export'] = bool(args.get('train_export', False))
        self.params['replace_features_use_new_name'] = True
        self.params['servingps_list'] = [
            {
                "cluster":
                    "zk://uf3001.zks.sina.com.cn:12185,uf3002.zks.sina.com.cn:12185,uf3003.zks.sina.com.cn:12185,uf3004.zks.sina.com.cn:12185,uf3005.zks.sina.com.cn:12185",
                "repo":
                    "weips-v6.3/mainpage-xbd-serving2/youfu"
            }
        ]
        if self.params['mode'] in ['train', 'export', 'deploy','predict']:
           self.params['ps_num'] = len(json.loads(os.environ.get("TF_CONFIG")).get('cluster').get('ps'))
        # 2、get sample address
        self.params['needOffline'] = args.get('needOffline', 'true')

        ## 2.1 hdfs
        self.params['sample_file_path'] = args.get('sample_path', '')

        self.params['item_embedding_path'] = args.get('item_embedding_path', '')

        # dt_partition 例如 20210617
        self.params['dt_partition'] = args.get('dt_partition').split(',') if args.get('dt_partition').lower() != 'none' \
                                                                   and args.get('dt_partition') else None
        dirs_pattern = 'dt={}'

        # 最小分区是否为hour分区
        self.params['is_hour_partition'] = args.get('is_hour_partition')

        # 当mode=predict时，数据的版本号从上游获取
        if(self.params['mode'] == 'predict'):
            self.params['dt_partition'] = get_params_info(self.params['instance_id'], 'var_data_day_version')
            self.params['data_hour_version'] = get_params_info(self.params['instance_id'], 'var_data_hour_version')
            dirs_patterns_array = [str.format(dirs_pattern, x) for x in self.params['dt_partition']]
            self.params['dirs_pattern'] = '|'.join(dirs_patterns_array) + '/hour=%s/.*' % self.params['data_hour_version']
        else:
            dirs_patterns_array = [str.format(dirs_pattern, x) for x in self.params['dt_partition']]
            self.params['dirs_pattern'] = '|'.join(dirs_patterns_array) + '/hour=.*/.*'

        # topic
        self.params["sample_topic"] = args.get('sample_topic', 'super-topic-ulike-realread-realtime-sample-flat')
        self.params['sample_groupid'] = args.get('sample_groupid', 'wb_oprd_supertopic_algo_kz0ssr')
        self.params["sample_group_user"] = args.get('sample_group_user', 'wb_oprd_supertopic_algo')
        self.params["sample_group_password"] = args.get('sample_group_password', 'KVw2eKSU7YUiFe')
        self.params['partitions'] = args.get('partitions', 50)
        self.params['bootstrap_server'] = args.get('bootstrap_server', 'kfk30.c9.al.sina.com.cn:9110,kfk25.c9.al.sina.com.cn:9110,kfk33.c9.al.sina.com.cn:9110,kfk27.c9.al.sina.com.cn:9110,kfk03.c9.al.sina.com.cn:9110')


        # 3、get model params

        self.params['runtime_batchsize'] = int(args.get('batch_size', 1024))
        self.params['embed_dim'] = int(args.get('embed_dim', 64))
        self.params['l2_reg'] = float(args.get('l2_reg', 0.001))
        self.params['layer_nodes'] = [int(num) for num in args.get('layer_nodes').split(',')]
        self.params['active_fn'] = args.get('active_fn').split(',')
        self.params['learning_rate'] = float(args.get('learning_rate', 0.01))
        self.params['norm'] = args.get('norm')
        self.params['dropout_rate'] = float(args.get('dropout_rate'))
        self.params['temperature_weight'] = float(args.get('temperature_weight', '20.0'))
        self.params['loss'] = args.get('loss')
        self.params['max_steps'] = int(args.get('max_steps')) if args.get('max_steps') else None
        self.params['user_id'] = args.get('user_id')
        self.params['item_id'] = args.get('item_id')
        self.params['label'] = args.get('label')
        self.params['no_example'] = True


        # senet
        self.params['senet'] = True if args.get('senet', 'false') == 'true' else False


        # get schema
        self.params["feature_columns"] = dataschema.column_list
        self.params['sample_data_schema'] = dataschema.get_data_schema()
        self.params['sample_label_column'] = args.get('label')
        self.params['predict_columns'] = dataschema.get_prediction_schema()

       # for waic
        if os.environ.get('USER'):
            self.user = os.environ.get('USER')
        else:
            self.user = args.get('user')
        self.params['user'] = self.user


        print('--------------------------------------------------开始打印接收到的参数----------------------------------------')
        for key in self.params.keys():
            print(key, self.params.get(key))
        print('--------------------------------------------------打印接收到的参数结束----------------------------------------')


    def process(self, params):
        if params is None:
            params = self.params
        else:
            params.update(self.params)
        mode = params['mode']

        xbd.init_v2(self.params['model_name'], 'train', async_push_pull=True)

        # sample file
        file_path = params['sample_file_path']

        print('-------------------------------判断mode类型------------------------------')
        print('mode', mode)
        if mode == 'train':

            params['replace_features_use_new_name'] = False
            if 'TF_CONFIG' in os.environ or params['runtime_stream_eval'] is True:
                params['runtime_stream_eval'] = True
                params['runtime_stream_eval_dir'] = params['base_dir'] + '/eval'
                params['runtime_stream_eval_device'] = '/job:xevaluator/task:0'

            if params['needOffline']:
                params["sample_addr"] = DataAddr(
                source_type=SourceType.FILE,
                file_path=file_path,
                 compression_type='GZIP',
                dirs_pattern=params['dirs_pattern'],
                )
            else:
                print('-------dataAddr-------')
                print(self.params)

                params["sample_addr"] = DataAddr(
                    source_type=SourceType.KAFKA,
                    bootstrap_server=params['bootstrap_server'],
                    topics=params['sample_topic'],
                    partitions=params['partitions'],
                    offset="-1",
                    groupid=params['sample_groupid'],
                    username=params['sample_group_user'],
                    password=params['sample_group_password'])

        if mode == 'predict':
            params['replace_features_use_new_name'] = False
            params["sample_addr"] = DataAddr(
                source_type=SourceType.FILE,
                file_path=file_path,
                compression_type='GZIP',
                dirs_pattern=params['dirs_pattern'],
            )
        elif mode == 'deploy':
            params["sample_addr"] = DataAddr(
                source_type=SourceType.FILE,
                file_path=file_path,
                dirs_pattern=params['dirs_pattern'],
            )
        elif mode == 'eval':
            params["sample_addr"] = DataAddr(
                source_type=SourceType.FILE,
                file_path=file_path,
                dirs_pattern=params['dirs_pattern'],
            )
        elif mode == 'export':
            params['runtime_stream_eval'] = False
            params["sample_addr"] = DataAddr(source_type=SourceType.FILE,
                                             file_path=file_path,
                                             compression_type='GZIP',
                                             dirs_pattern=params['dirs_pattern'],
                                             )
        runtime = xRuntime(params, mode)
        runtime.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_id', type=str, default='')
    parser.add_argument('--model_id', type=str, default='')
    parser.add_argument('--user', type=str, default='chenxin10')
    parser.add_argument('--sample_path', type=str, default='')
    parser.add_argument('--item_embedding_path', type=str, default='')
    parser.add_argument('--is_hour_partition', type=bool, default=False)
    parser.add_argument('--dt_partition', type=str, default='')
    parser.add_argument('--user_id', type=str, default='')
    parser.add_argument('--item_id', type=str, default='')
    parser.add_argument('--active_fn', type=str, default='relu,relu,tanh')
    parser.add_argument('--loss', type=str, default='log_loss')
    parser.add_argument('--layer_nodes', type=str, default='256,128,64')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--label', type=str, default="")
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--needOffline", type=bool, default=False)



    args = parser.parse_args()
    print(args)

    XBDProcess({'instance_id': args.instance_id,
                'model_id': args.model_id,
                'user': args.user,
                'sample_path': args.sample_path,
                'item_embedding_path': args.item_embedding_path,
                'is_hour_partition': args.is_hour_partition,
                'dt_partition': args.dt_partition,
                'user_id': args.user_id,
                'item_id': args.item_id,
                'active_fn': args.active_fn,
                'loss': args.loss,
                'layer_nodes': args.layer_nodes,
                'dropout_rate': args.dropout_rate,
                'batch_size': args.batch_size,
                'l2_reg': args.l2_reg,
                'label': args.label,
                'embed_dim': args.embed_dim,
                'learning_rate': args.learning_rate,
                'mode': args.mode,
                'needOffline': args.needOffline,
                }).process(None)




