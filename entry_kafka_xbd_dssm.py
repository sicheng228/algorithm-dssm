# coding=utf-8
from optparse import OptionParser
import argparse
import os
import tensorflow as tf
import sys
import logging
import xembedding as xbd

sys.path.append("../../")
from xembedding.source.BaseSource import *
from xembedding.source.KafkaSource import KafkaSource
from xembedding.source.FileSource import FileSource
from xembedding.model.BaseModel import GenFeatureConfHook,MetricsType, XBDPredictor, MetricsCalculator
from model_dssm_xbd import DSSM
from feature import *


# logger = logging.getLogger('tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
# formatter = logging.Formatter(
#     '%(asctime)s - %(filename)s - %(levelname)s:%(message)s')
# for h in logger.handlers:
#     h.setFormatter(formatter)
tf.logging.set_verbosity(tf.logging.INFO)
pid = os.getpid()
print("<<<<<<<<<<<START<<<<XBD PID:%d" % (pid))


class xRuntime():
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        stream_evaluator = None
        if 'runtime_stream_eval' in params and params['runtime_stream_eval'] is True:
            stream_evaluator = xbd.StreammingEvaluator(
                eval_dir=params['runtime_stream_eval_dir'],
                batch_size=params['runtime_batchsize'],
                dequeue_cnt=100)

            device_filters = stream_evaluator.get_device_filters()
            self.params['runtime_evaluator'] = stream_evaluator
        else:
            device_filters = self.default_device_filter()

        # If the task_type is `EVALUATOR` or something other than the ones in
        # TaskType then don't set any device filters.
        # device_filters = None
        session_config = tf.ConfigProto(allow_soft_placement=True, device_filters=device_filters)
        self.run_config = tf.estimator.RunConfig(keep_checkpoint_max=10,
                                                 log_step_count_steps=100,
                                                 session_config=session_config)
        params['is_chief'] = self.run_config.is_chief
        if params['sample_data_schema'] is not None:
            if 'predict_offline' in params and params['predict_offline'] is True:
                self.source = FileSource(
                    batchsize=params['runtime_batchsize'],
                    label_column=params['sample_label_column'] if mode == 'train' else None,
                    sample_addr=params['sample_addr'],
                    sample_schema=params['sample_data_schema'],
                    need_split=False,
                    train_data_ratio=0.9,
                    worker_num=self.run_config.num_worker_replicas if 'TF_CONFIG' in os.environ else 1,
                    worker_index=self.run_config.global_id_in_cluster if 'TF_CONFIG' in os.environ else 0,
                    decode_op='xbd',
                    map_parallel=20,
                    prefetch_size=params['runtime_batchsize']*100,
                    use_quote_delim=False)
            else:
                self.source = KafkaSource(
                    batchsize=params['runtime_batchsize'],
                    label_column=params['sample_label_column'],
                    sample_addr=params['sample_addr'],
                    sample_schema=params['sample_data_schema'],
                    shuffle_size=False,
                    need_split=False,
                    train_data_ratio=0.9,
                    worker_num=self.run_config.num_worker_replicas if 'TF_CONFIG' in os.environ else 1,
                    worker_index=self.run_config.global_id_in_cluster if 'TF_CONFIG' in os.environ else 0,
                    decode_op='xbd',
                    map_parallel=20,
                    prefetch_size=params['runtime_batchsize']*100,
                    use_quote_delim=False)

        self.model = DSSM(params=params)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model.get_model_fn(),
            config=self.run_config,
            model_dir=params['model_dir'],
            params=params)

    def default_device_filter(self):
        run_config_tmp = tf.estimator.RunConfig()
        device_filters = None
        debug_info = "If the task_type is `EVALUATOR` or something other than the ones in " \
                     "TaskType then don't set any device filters."

        if run_config_tmp.task_type == 'master':
            return ['/job:ps', '/job:master']
        elif run_config_tmp.task_type == 'chief':
            return ['/job:ps', '/job:chief']
        elif run_config_tmp.task_type == 'worker':
            return ['/job:ps', '/job:worker/task:%d' % run_config_tmp.task_id]
        elif run_config_tmp.task_type == 'ps':
            return ['/job:ps', '/job:worker', '/job:chief', '/job:master']
        else:
            print(debug_info)
        return device_filters

    def train(self):
        self.estimator.train(input_fn=lambda: self.source.get_input_fn(), steps=None)

    def train_and_eval(self):
        hooks = []
        # notice chief last exit hook:
        # must add before CheckpointSaverHook
        # CheckpointSaverHook must added as normal hook
        chief_last_exit_hook = xbd.ChiefLastExitHook(self.run_config.num_worker_replicas, self.params['is_chief'])
        hooks.append(chief_last_exit_hook)

        if self.params['is_chief'] is True:
            chief_saver_hook = xbd.CheckpointSaverHook(
                checkpoint_dir=self.params['model_dir'],
                save_secs=900,
                checkpoint_basename='model.ckpt',
                need_checkpoint_loader=True,
                keep_weips_checkpoint_max=5)

            hooks.append(chief_saver_hook)

        train_spec = tf.estimator.TrainSpec(input_fn=lambda: self.source.get_input_fn(), hooks=hooks)

        exporter = tf.estimator.LatestExporter('exporter_supertopic_prerank_dssm', self.model.get_serving_input_fn(serving=True, raw_feature_tensor=False),
                                               as_text=False, exports_to_keep=5)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.source.get_input_fn(),  # no need to batch in eval
            steps=10, start_delay_secs=20,  # start evaluating after N seconds
            exporters=exporter, throttle_secs=60,  # evaluate every N seconds
        )
        print("source get input fn: location 138", self.source.get_input_fn().__dict__)
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)


    def export(self):
        self.estimator.export_saved_model(as_text=True,
                                          export_dir_base=self.params['model_export_dir'],
                                          serving_input_receiver_fn=self.model.get_serving_input_fn())

        a = GenFeatureConfHook([self.params['model_root_dir'],
                                os.path.join(self.params['model_root_dir'],
                                             'model_version=%s' % self.params['model_version'])],
                               feature_spec=self.model.get_serving_input_fn(False)())
        a.after_create_session("", "")

    def eval(self):
        self.estimator.evaluate(input_fn=lambda: self.source.get_input_fn(),
                                steps=None, hooks=[], name='test_dataset_eval')

    def predict(self):
        """
       XBDPredictor 用以预测样本，并写到预测文件中，支持分布式和单机两种方式，支持xbd模型(包含dense/sparse部分)以及tf模型(只有dense部分),
       如果是xbd模型，需要事先调用xbd.init，mode参数还是填写"train",
       但其中的repo需要填评估weips地址，不要写训练weips地址, 不然每次评估的结果可能不一样，并且训练ps上的模型会被之前保存的快照覆盖！！！
       XBDPredictor 构造函数参数说明:
       run_config: 运行时配置
       estimator: tf.estimator
       XBDPredictor predict 函数参数说明:
       predict_version: 保存快照对应的global step, predict_version为None则是使用最新快照进行评估。
       predict_version可以查看模型快照路径下的checkpoint文件或者xbd_checkpoint文件获得，如果是xbd模型的话，还可以搜寻训练chief的日志，通过查找"display xbd checkpoint" 查看有哪些快照
       predict_keys: 预测文件中的字段
       predict_result_path: 预测文件存储的目录，一般是checkpoint目录下的子目录，该目录在预测开始的时候会进行删除，切记不要写成重要的目录，防止误删！！！

       MetricsCalculator 用来使用预测文件计算auc或者gauc
       MetricsCalculator 构造函数参数说明:
       metrics_type: 类型，gauc or auc
       path: 与XBDPredictor predict函数的predict_result_path一致
       predit_keys: 与XBDPredictor predict函数的predict_keys一致
       label，prob: 标签与概率，必填
       group_key, filter_key, weight_key: 计算gauc必填
       MetricsCalculator calculate说明:
       计算auc或者gauc，并输出到标准输出
         """

        out_fields = ['item_id', 'item_vector']

        # metrics_calculator = MetricsCalculator(
        #     metrics_type=MetricsType.GAUC,
        #     path=self.params['predict_path'],
        #     predict_keys=out_fields,
        #     group_key="fm_mid",
        #     filter_key="fu_id",
        #     weight_key="fm_mid",
        #     label="label",
        #     prob="probabilities",
        #     chief=self.run_config.is_chief
        # )


        # predictor = XBDPredictor(self.run_config, self.estimator)
        predictor = XBDPredictor(self.run_config, self.estimator, model_id=self.params['model_name'],
                                 total_num=100000,
                                 batch=self.params['runtime_batchsize'])
        predictor.predict(input_fn=lambda: self.source.get_input_fn(), predict_version=None,
                          predict_keys=out_fields, predict_result_path=self.params['predict_path'], func_type='json')
        # metrics_calculator.calculate()


    def run(self):
        if self.mode == "train":
            if 'runtime_evaluator' in self.params and self.params['runtime_evaluator'] is not None:
                my_evaluator = self.params['runtime_evaluator']
                exporter = xbd.XBDExporter(
                    estimator=self.estimator,
                    model_dir=self.params['model_dir'],
                    export_dir=self.params['model_export_dir'],
                    serving_input_receiver_fn=self.model.get_serving_input_fn(),
                    feature_spec=None if self.params['no_example'] else self.model.get_serving_input_fn(
                        serving=False)(),
                    receiver_tensor=self.model.get_serving_input_fn(serving=False, raw_feature_tensor=True)() if
                    self.params['no_example'] else None,
                    keep_max_savedmodel_num=1, )
                if self.params['model_deploy'] is True:
                    exporter.set_deploy(ps_list=self.params["servingps_list"], model_name=self.params["model_name"])
                my_evaluator.set_export(exporter)
                my_evaluator.start()
            if self.params['train']:
                self.train_and_eval()

        elif self.mode == "eval":
            self.eval()
        elif self.mode == "predict":
            self.predict()
        elif self.mode == "export":
            self.export()