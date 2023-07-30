# coding=utf-8
from optparse import OptionParser
import argparse
import os,time,json,re
import tensorflow as tf
import sys
import numpy as np
import logging
import xembedding as xbd
import dataschema
from utils.helios_api  import *

sys.path.append("../../")
from xembedding.source.BaseSource import *
from xembedding.source.FileSource import FileSource
from xembedding.model.BaseModel import GenFeatureConfHook,MetricsType, XBDPredictor, MetricsCalculator
from model_dssm_xbd import DSSM

logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
formatter = logging.Formatter(
    '%(asctime)s - %(filename)s - %(levelname)s:%(message)s')
for h in logger.handlers:
    h.setFormatter(formatter)
tf.logging.set_verbosity(tf.logging.INFO)
pid = os.getpid()
print("<<<<<<<<<<<START<<<<XBD PID:%d" % (pid))


class xRuntime():
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        stream_evaluator = None
        device_filters = None
        if 'runtime_stream_eval' in params and params['runtime_stream_eval'] is True:
            stream_evaluator = xbd.StreammingEvaluator(
                eval_dir=params['runtime_stream_eval_dir'],
                batch_size=params['runtime_batchsize'],
                dequeue_cnt=50)
            device_filters = stream_evaluator.get_device_filters()
            self.params['runtime_evaluator'] = stream_evaluator
        else:
            device_filters = self.default_device_filter()

        # If the task_type is `EVALUATOR` or something other than the ones in
        # TaskType then don't set any device filters.
        # device_filters = None
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            # log_device_placement=True,
            device_filters=device_filters)

        self.run_config = tf.estimator.RunConfig(keep_checkpoint_max=5,
                                                 log_step_count_steps=100,
                                                 session_config=session_config)
        print('run_config', self.run_config)

        params['is_chief'] = self.run_config.is_chief
        print("---------------------------------------file source set---------------------------------------")
        if params['sample_data_schema'] is not None:
            if 'TF_CONFIG' in os.environ:
                self.source = FileSource(
                    batchsize=params['runtime_batchsize'],
                    label_column=params['sample_label_column'] if mode == 'train' else None,
                    sample_addr=params['sample_addr'],
                    sample_schema=params['sample_data_schema'],
                    shuffle_size=10000,
                    need_split=False,
                    train_data_ratio=0.9,
                    worker_num=self.run_config.num_worker_replicas,
                    worker_index=self.run_config.global_id_in_cluster,
                    decode_op='xbd',
                    map_parallel=20,
                    prefetch_size=params['runtime_batchsize']*100,
                    use_quote_delim=False)
            else:
                print(" now TF_CONFIG not find ")
                self.source = FileSource(
                    batchsize=params['runtime_batchsize'],
                    label_column=params['sample_label_column'] if mode == 'train' else None,
                    sample_addr=params['sample_addr'],
                    sample_schema=params['sample_data_schema'],
                    shuffle_size=False,
                    need_split=False,
                    train_data_ratio=0.9,
                    decode_op='xbd',
                    map_parallel=20,
                    prefetch_size=params['runtime_batchsize'] * 100,
                    use_quote_delim=False)

        # set model
        self.model = DSSM(params=params)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model.get_model_fn(),
            config=self.run_config,
            model_dir=params['model_dir'],
            params=params)

    def get_model_version(self):
            count = 0
            while True:
                time.sleep(5)
                dirs = [int(i) for i in tf.gfile.ListDirectory(self.params['model_export_dir']) if
                        re.match('^[0-9]+$', i)]
                if len(dirs) > 0:
                    break
                if count > 2:
                    raise ValueError('Export dir is empty, has tried 3 times.')
                count += 1
            dirs.sort(reverse=True)
            export_version = str(dirs[0])
            return export_version

    def default_device_filter(self):
        run_config_tmp = tf.estimator.RunConfig()
        device_filters = None

        # Master should only communicate with itself and ps
        # chief should only communicate with itself and ps
        # Worker should only communicate with itself and ps
        if run_config_tmp.task_type == 'master':
            device_filters = ['/job:ps', '/job:master']
        elif run_config_tmp.task_type == 'chief':
            device_filters = ['/job:ps', '/job:chief']
        elif run_config_tmp.task_type == 'worker':
            device_filters = [
                '/job:ps',
                '/job:worker/task:%d' % run_config_tmp.task_id
            ]
        elif run_config_tmp.task_type == 'ps':
            device_filters = [
                '/job:ps', '/job:worker', '/job:chief', '/job:master'
            ]
        else:
            print(
                "If the task_type is `EVALUATOR` or something other than the ones in TaskType then don't set any "
                "device filters. "
            )
        return device_filters

    def train(self):
        self.estimator.train(input_fn=lambda: self.source.get_input_fn(), steps=None)

    def train_and_eval(self):
        hooks = []
        # notice chief last exit hook:
        # must add before CheckpointSaverHook
        # CheckpointSaverHook must added as normal hook
        chief_last_exit_hook = xbd.ChiefLastExitHook(
            self.run_config.num_worker_replicas, self.params['is_chief'])

        hooks.append(chief_last_exit_hook)
        if self.params['is_chief'] is True:
            feature_conf_dir = os.path.join(
                self.params['model_root_dir'],
                'model_version=%s' % self.params['model_version'])

            print("feature_conf_dir:%s" % (feature_conf_dir))

            gen_feature_hook = xbd.GenFeatureConfHook(
                path=feature_conf_dir,
                feature_spec=self.model.get_serving_input_fn(serving=False, raw_feature_tensor=False)())

            chief_saver_hook = xbd.CheckpointSaverHook(
                checkpoint_dir=self.params['model_dir'],
                save_steps=50000,
                checkpoint_basename='model.ckpt',
                need_checkpoint_loader=True,
                keep_weips_checkpoint_max=2)
            profiler_hook = tf.train.ProfilerHook(
                save_secs=1200,
                output_dir=self.params['model_profile_dir'],
                show_dataflow=True,
                show_memory=False)

            # hooks.append(profiler_hook)
            hooks.append(gen_feature_hook)
            hooks.append(chief_saver_hook)

        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.source.get_input_fn(),
            max_steps=self.params['max_steps'],
            hooks=hooks)

        exporter = tf.estimator.LatestExporter(
            'exporter_super_topic',
            self.model.get_serving_input_fn(),
            as_text=False,
            exports_to_keep=3)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.source.get_input_fn(
            ),  # no need to batch in eval
            steps=10,
            start_delay_secs=20,  # start evaluating after N seconds
            throttle_secs=60,  # evaluate every N seconds
            exporters=exporter,
        )
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)


    def predict(self):

        embedding_path = os.path.join(self.params['item_embedding_path'], self.params['dt_partition'], self.params['data_hour_version'])

        print("embedding_path: %s" % embedding_path)

        columns = ['item_id', 'item_vector']

        predictor = XBDPredictor(self.run_config, self.estimator, model_id=self.params['model_name'],
                                 total_num=2000000,
                                 batch=15)
        predictor.predict(input_fn=lambda: self.source.get_input_fn(), predict_version=None,
                          predict_keys=columns, predict_result_path=embedding_path, func_type='default')


    def export(self, dir=None):

        print('-------------------------进入export----------------------')

        self.estimator.export_saved_model(
            export_dir_base=self.params['model_export_dir'],
            serving_input_receiver_fn=self.model.get_serving_input_fn(serving=True, raw_feature_tensor=False),
            as_text=False)

        print('-------------------------copy model to new path----------------------')

        #找到export_dir下的最新的时间戳
        count = 0
        while True:
            time.sleep(5)
            dirs = [int(i) for i in tf.gfile.ListDirectory(self.params['model_export_dir']) if re.match('^[0-9]+$', i)]
            if len(dirs) > 0:
                break
            if count > 2:
                raise ValueError('Export dir is empty, has tried 3 times.')
            count += 1
        dirs.sort(reverse=True)
        export_version = str(dirs[0])

        print('export_version', export_version)

        a = GenFeatureConfHook([dir if dir else self.params['model_root_dir'],
                                os.path.join(dir if dir else self.params['model_root_dir'],
                                             'model_version=%s' % export_version)],
                               feature_spec=self.model.get_serving_input_fn()())
        a.after_create_session("", "")

        model_version_path = os.path.join(self.params['model_root_dir'], "model_version=" + export_version)

        if tf.gfile.Exists(model_version_path):
            tf.gfile.DeleteRecursively(model_version_path)
        else:
            tf.gfile.MakeDirs(model_version_path)

        source_dir = os.path.join(self.params['model_export_dir'], export_version)
        tf.gfile.Copy(os.path.join(source_dir, 'saved_model.pb'), os.path.join(model_version_path, 'saved_model.pb'), overwrite=True)
        for root, dirs, files in tf.gfile.Walk(os.path.join(source_dir, 'variables')):
            for name in files:
                tf.gfile.Copy(os.path.join(source_dir, 'variables', name), os.path.join(model_version_path, 'variables', name), overwrite=True)

        print("SINASCHEDULERPARAMGEN:model_version="+export_version)
        sys.stdout.flush()

        data = {'var_model_version': export_version}

        return return_info_to_helios(self.params['instance_id'], data)


    def eval(self):
        hooks = []
        # hooks.append(xbd.RegisterEmbeddingHook())
        # hooks.append(xbd.InitEmbeddingHook())
        self.estimator.evaluate(input_fn=lambda: self.source.get_input_fn(),
                                steps=None,
                                hooks=hooks,
                                name='test_dataset_eval')

    def run(self):
        print("++++++++++++IN HDFS MODE+++++++++++")
        if self.mode == "train":

            if self.params['train']:
                self.train_and_eval()

            if 'runtime_hdfs_eval' in self.params and self.params['runtime_hdfs_eval'] is True and self.run_config.is_chief:
                print("+++++++++++++++runtime_evaluator in here+++++++++++++++")

            if 'runtime_evaluator' in self.params and self.params['runtime_evaluator'] is not None:
                my_evaluator = self.params['runtime_evaluator']

                exporter = xbd.XBDExporter(estimator=self.estimator,
                                         model_dir=self.params['model_dir'],
                                         export_dir=self.params['model_export_dir'],
                                         serving_input_receiver_fn=self.model.get_serving_input_fn(
                                             serving=True, raw_feature_tensor=True),
                                         receiver_tensor=self.model.get_serving_input_fn(
                                             serving=False, raw_feature_tensor=True)(),
                                         keep_max_savedmodel_num=3,
                                         feature_spec=None)

                if self.params['model_deploy'] is True:
                    exporter.set_deploy(ps_list=self.params["servingps_list"], model_name=self.params["model_name"])

                my_evaluator.set_export(exporter)
                my_evaluator.start()

            if self.params['train_export'] and self.run_config.is_chief:
                # export for
                export = xbd.XBDExporter(estimator=self.estimator,
                                         model_dir=self.params['model_dir'],
                                         export_dir=self.params['model_export_dir'],
                                         serving_input_receiver_fn=self.model.get_serving_input_fn(
                                             serving=True, raw_feature_tensor=True),
                                         receiver_tensor=self.model.get_serving_input_fn(
                                             serving=False, raw_feature_tensor=True)(),
                                         keep_max_savedmodel_num=3,
                                         feature_spec=None)
                if self.params['model_deploy'] is True:
                    export.set_deploy(ps_list=self.params["servingps_list"], model_name=self.params["model_name"])
                export.export()

                # export for super topic
                if self.params['replace_features_use_new_name']:
                    self.export(self.params['super_topic_model_export_dir'])

        elif self.mode == "eval":
            self.eval()
        elif self.mode == "predict":
            self.predict()
        elif self.mode == "export":
           if self.run_config.is_chief:
              self.export()
        elif self.mode == 'deploy':
            if self.run_config.is_chief:
                export = xbd.XBDExporter(estimator=self.estimator,
                                         model_dir=self.params['model_dir'],
                                         export_dir=self.params['model_export_dir'],
                                         serving_input_receiver_fn=self.model.get_serving_input_fn(
                                             serving=True, raw_feature_tensor=True),
                                         receiver_tensor=self.model.get_serving_input_fn(
                                             serving=False, raw_feature_tensor=True)(),
                                         keep_max_savedmodel_num=3,
                                         feature_spec=None)
                export.set_deploy(ps_list=self.params["servingps_list"], model_name=self.params["model_name"])
                export.export()

        elif self.mode == 'test':
            dataset = self.source.get_input_fn()
            iter = dataset.make_one_shot_iterator()
            i = 0
            with tf.Session() as sess:
                while i < 10:
                    print(sess.run(iter.get_next()))
                    i += 1
