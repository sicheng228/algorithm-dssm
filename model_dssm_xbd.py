# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import logging
import time
import six
import os

from xembedding.model.BaseModel import *
from xembedding.feature.WeiFeature import *
from xembedding import StreammingEvaluator, StreammingEvaluatorV2
from tensorflow.python.ops import lookup_ops
from feature import feature_process
from tensorflow.python.feature_column.feature_column_v2 import *
from dataschema import *


class DSSM(BaseModel):
    def __init__(self, **kwargs):
        self.params = kwargs['params']
        self.is_sync = self.params.get('runtime_is_sync', False)
        print("dssm XXXXXXXXX:%d" % self.is_sync)

    def init(self):
        self.exception_list = [
            "null", "NULL", "error_1", "exception", "|", "-1"
        ]
        self.table = lookup_ops.index_table_from_tensor(
            vocabulary_list=self.exception_list, default_value=-1)

    def fcast(self, feature, dtype):
        # kick exception value
        cond = tf.equal(feature, '')
        # FIXBUG: tf.shape(feature) will be error when get file end
        cond = tf.where(
            cond, tf.tile(tf.constant([False]), tf.shape(cond)),
            tf.equal(
                self.table.lookup(feature),
                tf.tile(tf.constant([-1], dtype=tf.int64), tf.shape(cond))))
        num_str = tf.where(cond, feature, tf.tile(["0"], tf.shape(cond)))

        return tf.string_to_number(num_str, out_type=dtype)

    def label_cast(self, labels):
        return self.fcast(labels, tf.float32)


    def model_fn(self, features, labels, mode, params):

        self.init()

        if labels is not None: labels = self.fcast(labels, tf.float32)

        ### get necessary params
        layer_nodes = params['layer_nodes']
        active_fn = params['active_fn']
        layer_cnt = len(layer_nodes)
        l2_reg = params['l2_reg']
        norm = params['norm']
        dimension = params['embed_dim']
        learning_rate =params['learning_rate']
        ps_num = params['ps_num']

        print("-----------------------------------------特征处理------------------------------")


        # if labels is not None:
        #     labels = self.fcast(labels, tf.float32)

       ##### 1、feature process
        user_columns, item_columns, user_group_features, item_group_features = feature_process()

        user_feature =[xbd.feature_column.wei_embedding_column(user_group_features, mean=0, std=0.01, combiner='mean',
                                                    dimension=dimension,
                                                    learning_rate=learning_rate,
                                                    expired_min =60 * 24 * 7)]

        item_feature = [xbd.feature_column.wei_embedding_column(item_group_features, mean=0, std=0.01, combiner='mean',
                                                    dimension=dimension,
                                                    expired_min=60 * 24 * 7,
                                                    learning_rate=learning_rate)]

        # user feas
        # 1. user input [batch_size, feat_count*embed_size]
        print('features:', features)
        print('user_feature', user_feature)
        user_input = xbd.feature_column.input_layer(features, user_feature, trainable=True)

        # item feas
        # 1. item input [batch_size, feat_count*embed_size]
        item_input = xbd.feature_column.input_layer(features, item_feature, trainable=True)

        def SENET_layer(embedding_matrix, field_size, dimension, pool_op, ratio, ps_num):
            with tf.variable_scope('SENET_layer', partitioner=tf.fixed_size_partitioner(ps_num, axis=0)):
                # squeeze embedding to scaler for each field
                with tf.variable_scope('pooling', partitioner=tf.fixed_size_partitioner(ps_num, axis=0)):
                    if pool_op == 'max':
                        z = tf.reduce_max(embedding_matrix,
                                          axis=2)  # batch * field_size * emb_size -> batch * field_size
                    else:
                        z = tf.reduce_mean(embedding_matrix, axis=2)

                # excitation learn the weight of each field from above scaler
                with tf.variable_scope('excitation', partitioner=tf.fixed_size_partitioner(ps_num, axis=0)):
                    z1 = tf.layers.dense(z, kernel_initializer=tf.initializers.glorot_normal(),
                                         units=field_size // ratio, activation='relu')
                    a = tf.layers.dense(z1, kernel_initializer=tf.initializers.glorot_normal(), units=field_size,
                                        activation='relu')  # batch * field_size

                # re-weight embedding with weight
                with tf.variable_scope('reweight', partitioner=tf.fixed_size_partitioner(ps_num, axis=0)):
                    senet_embedding = tf.multiply(embedding_matrix,
                                                  tf.expand_dims(a,
                                                                 axis=-1))  # (batch * field * emb) * ( batch * field * 1)
                return senet_embedding

        # user view
        with tf.variable_scope("User_View", partitioner=tf.fixed_size_partitioner(ps_num, axis=0)):
            if params['senet']:
                field_size = np.int(user_input.get_shape().as_list()[1] / params['embed_dim'])
                user_input_reshape = tf.reshape(user_input, [-1, np.int(field_size), np.int(params['embed_dim'])])
                user_input_senet = SENET_layer(user_input_reshape, field_size, params['embed_dim'], 'max', 2,
                                               ps_num)
                user_input = tf.reshape(user_input_senet, [-1, np.int(field_size * params['embed_dim'])])
            else:
                field_size = len(user_columns)
                print("user-preshape:", user_input)
                user_input = tf.reshape(user_input, [-1, np.int(field_size * params['embed_dim'])])
                print("user-shape:", user_input)
                user_input = tf.layers.dropout(
                    user_input, rate=params['dropout_rate']
                ) if mode == tf.estimator.ModeKeys.TRAIN else user_input

            user_nodes = {-1: user_input}

            for i in range(layer_cnt):
                user_nodes[i] = tf.layers.dense(user_nodes[i - 1],
                                                kernel_initializer=tf.initializers.glorot_normal(),
                                                units=layer_nodes[i], activation=None,
                                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))
                if active_fn[i] == 'relu':
                    user_nodes[i] = tf.nn.relu(user_nodes[i])
                elif active_fn[i] == 'tanh':
                    user_nodes[i] = tf.nn.tanh(user_nodes[i])
                elif active_fn[i] == 'sigmoid':
                    user_nodes[i] = tf.nn.sigmoid(user_nodes[i])
                elif active_fn[i] == 'leak_relu':
                    user_nodes[i] = tf.nn.leaky_relu(user_nodes[i])

                # [2022.04.01 10:29]
                if norm == 'ln':
                    user_nodes[i] = tf.keras.layers.LayerNormalization(user_nodes[i])
                elif norm == 'bn':
                    user_nodes[i] = tf.layers.batch_normalization(user_nodes[i])
                elif norm == 'vo-ln':
                    _, var = tf.nn.moments(user_nodes[i], axes=[-1])
                    var = tf.expand_dims(var, -1)
                    user_nodes[i] = user_nodes[i] / (tf.sqrt(var) + 0.000001)

                if mode == tf.estimator.ModeKeys.TRAIN and i < layer_cnt - 1:
                    user_nodes[i] = tf.layers.dropout(user_nodes[i], rate=params['dropout_rate'])
                print("layers:", user_nodes[i])

            user_vector = tf.nn.l2_normalize(user_nodes[layer_cnt - 1], axis=1, name='user_embedding')

        # item view
        with tf.variable_scope('Item_View', partitioner=tf.fixed_size_partitioner(ps_num, axis=0)):
            if params['senet']:
                field_size = np.int(item_input.get_shape().as_list()[1] / params['embed_dim'])
                item_input_reshape = tf.reshape(item_input, [-1, np.int(field_size), np.int(params['embed_dim'])])
                item_input_senet = SENET_layer(item_input_reshape, field_size, params['embed_dim'], 'max', 2,
                                               ps_num)
                item_input = tf.reshape(item_input_senet, [-1, np.int(field_size * params['embed_dim'])])
            else:
                print("item-preshape:", item_input)
                field_size = len(item_columns)
                item_input = tf.reshape(item_input, [-1, np.int(field_size * params['embed_dim'])])
                print("item-shape:", item_input)
                item_input = tf.layers.dropout(
                    item_input, rate=params['dropout_rate']
                ) if mode == tf.estimator.ModeKeys.TRAIN else item_input

            item_nodes = {-1: item_input}

            for i in range(layer_cnt):
                item_nodes[i] = tf.layers.dense(item_nodes[i - 1],
                                                kernel_initializer=tf.initializers.glorot_normal(),
                                                units=layer_nodes[i], activation=None,
                                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))

                if active_fn[i] == 'relu':
                    item_nodes[i] = tf.nn.relu(item_nodes[i])
                elif active_fn[i] == 'tanh':
                    item_nodes[i] = tf.nn.tanh(item_nodes[i])
                elif active_fn[i] == 'sigmoid':
                    item_nodes[i] = tf.nn.sigmoid(item_nodes[i])
                elif active_fn[i] == 'leak_relu':
                    item_nodes[i] = tf.nn.leaky_relu(item_nodes[i])

                if norm == 'ln':
                    item_nodes[i] = tf.keras.layers.LayerNormalization(item_nodes[i])
                elif norm == 'bn':
                    item_nodes[i] = tf.layers.batch_normalization(item_nodes[i])
                elif norm == 'vo-ln':
                    _, var = tf.nn.moments(item_nodes[i], axes=[-1])
                    var = tf.expand_dims(var, -1)
                    item_nodes[i] = item_nodes[i] / (tf.sqrt(var) + 0.000001)

                if mode == tf.estimator.ModeKeys.TRAIN and i < layer_cnt - 1:
                    item_nodes[i] = tf.layers.dropout(item_nodes[i], rate=params['dropout_rate'])
                print("layers:", user_nodes[i])
            item_vector = tf.nn.l2_normalize(item_nodes[layer_cnt - 1], axis=1, name='item_embedding')

        y_1 = tf.multiply(user_vector, item_vector)
        y_2 = tf.reduce_sum(y_1, axis=1)
        y = tf.sigmoid(params['temperature_weight'] * y_2, name='pred_score')

        if mode == tf.estimator.ModeKeys.PREDICT:
            print('---------------------predict------------------------------')
            print("label", params.get('label'))
            print("label_value", features.get(params.get('label')))

            # user_id = tf.identity(features.get(params.get('user_id')), name='user_id')
            # item_id = tf.identity(features.get(params.get('item_id')), name='item_id')
            # label = tf.identity(features.get(params.get('label')), name='label')
            # predictions = {'user_id': user_id, 'scores': user_vector,
            #                'item_id': item_id, 'item_vector': item_vector,
            #                'label': label, 'pred_score': y}
            predictions = {'scores': user_vector,
                           'item_vector': item_vector,
                           'pred_score': y}
            if params['mode'] == 'predict':
                item_id = tf.identity(features.get(params.get('item_id')), name='item_id')
                predictions = {'scores': item_vector,
                               'item_vector': item_vector,
                               'pred_score': y,
                               'item_id': item_id}
            export_outputs = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                    predictions)}

            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs=export_outputs,
                prediction_hooks=[xbd.InitEmbeddingHook()])

       # loss
        if params['loss'] == 'mean_squared_error':
            labels = params['factor'] * labels
            loss = tf.losses.mean_squared_error(labels=labels, predictions=y)
        elif params['loss'] == 'weighted_cross_entropy_with_logits':
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=y, pos_weight=100)
        else:
            loss = tf.losses.log_loss(labels=labels, predictions=y)
        loss_avg = tf.reduce_mean(loss, name='loss_avg')

        pos_num = tf.reduce_sum(labels)
        zero_labels = tf.zeros_like(labels)
        ones_labels = tf.ones_like(labels)
        neg_num = tf.reduce_sum(tf.where(tf.equal(zero_labels, labels), ones_labels, zero_labels))
        # click mean score
        click_index = tf.where(labels)
        pred_click_score = tf.gather_nd(y, click_index)
        predict_pos_score = tf.reduce_mean(pred_click_score)
        # not click mean score
        not_click = tf.zeros_like(labels)
        not_click_matrix = tf.equal(labels, not_click)
        not_click_index = tf.where(not_click_matrix)
        pred_not_click_score = tf.gather_nd(y, not_click_index)
        predict_neg_score = tf.reduce_mean(pred_not_click_score)

        # user_id = tf.identity(features.get(params.get('user_id')), name='user_id')
        # item_id = tf.identity(features.get(params.get('item_id')), name='item_id')

        def eval_fn():
            predicted_classes = tf.cast(y > 0.5, tf.int32)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='accuracy')
            auc = tf.metrics.auc(labels=labels, predictions=y, name='auc')
            metrics = {'accuracy': accuracy, 'auc': auc}
            return accuracy, auc, metrics

        if params['loss'] == "mean_squared_error":
            mse = tf.metrics.mean_squared_error(labels=labels, predictions=y, name='mse')
            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {'mse': mse}
                return tf.estimator.EstimatorSpec(mode, loss=loss_avg, eval_metric_ops=metrics)

            logging_hook = tf.train.LoggingTensorHook({'pos_num': pos_num, "mse": mse[1],
                                                       'predict_pos_socre': predict_pos_score,
                                                       'predict_neg_score': predict_neg_score}, every_n_iter=100)

        else:

            accuracy, auc, metrics = eval_fn()


            if mode == tf.estimator.ModeKeys.EVAL:
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('auc', auc)
                tf.summary.scalar('loss_avg', loss_avg)
                return tf.estimator.EstimatorSpec(mode, loss=loss_avg, eval_metric_ops=metrics, evaluation_hooks=[
                                  xbd.InitEmbeddingHook()])

            tf.summary.scalar('auc', auc[1])
            tf.summary.scalar('loss', loss_avg)
            logging_hook = tf.train.LoggingTensorHook({'pos_num': pos_num, 'neg_num': neg_num,
                                                       "auc": auc[1], 'accuracy': accuracy[1],
                                                       'predict_pos_socre': predict_pos_score,
                                                       'predict_neg_score': predict_neg_score,
                                                       'loss': loss_avg,
                                                       # 'uid': user_id,
                                                       # 'user_input': user_input,
                                                       # 'user_nodes[0]': user_nodes[0],
                                                       # 'user_nodes[1]': user_nodes[1],
                                                       # 'user_nodes[2]': user_nodes[2],
                                                       # 'user_vector': user_vector,
                                                       # 'item_id': item_id,
                                                       # 'item_input': item_input,
                                                       # 'item_nodes[0]': item_nodes[0],
                                                       # 'item_nodes[1]': item_nodes[1],
                                                       # 'item_nodes[2]': item_nodes[2],
                                                       # 'item_vector': item_vector,
                                                       # 'y_1': y_1,
                                                       # 'y_2': y_2,
                                                       # 'y': y
                                                       }, every_n_iter=100)

        assert mode == tf.estimator.ModeKeys.TRAIN
        accuracy, auc, metrics = eval_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(accuracy[1])
        update_ops.append(auc[1])
        logging_hook = tf.train.LoggingTensorHook(
            {
                "loss": loss,
                "global_step": tf.train.get_global_step(),
                "pos_num": pos_num,
                'auc': auc[0],
            },
            every_n_iter=100,
        )
        training_hooks = []
        global_step = tf.train.get_global_step()
        decay_rate = 0.99
        decay_steps = 1000
        decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                           decay_steps=decay_steps,
                                                           decay_rate=decay_rate)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=learning_rate)
            if self.is_sync is True:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=3, total_num_replicas=3)
                sync_replicas_hook = optimizer.make_session_run_hook(
                    self.params['is_chief'])
                training_hooks.append(sync_replicas_hook)
            optimizer = xbd.DistributedOptimizer(optimizer)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, 100)
            train_op = optimizer.apply_gradients(
                zip(gradients, variables),
                global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss_avg, train_op=train_op, training_hooks=[logging_hook, xbd.InitEmbeddingHook()])


    def get_model_fn(self):
        return self.model_fn

    def get_log_formatter(self):
        def formatter(tensors):
            """
            Format the log output
            """
            logstring = "step {}:input:{},{} vs {},loss:{} \
                       ".format(tensors["global_step"], tensors["features"],
                                tensors["y_real"][0], tensors["y_hat"][0],
                                tensors["loss"])
            return logstring

        return formatter

    def get_serving_input_fn(self, serving=True, raw_feature_tensor=False):
        def serving_input_receiver_fn():
            # https://zhuanlan.zhihu.com/p/46926928
            # """An input receiver that expects a serialized tf.Example."""

            feature_spec = {}
            user_columns, item_columns, user_group_features, item_group_features = feature_process()
            self.params['feature_columns'] = user_columns + item_columns

            for column in self.params["feature_columns"]:
                column = xbd.get_feature_name_v2(column)
                feature_spec[column] = tf.FixedLenFeature([], dtype=tf.string)

            print("xxxserving_input_reciver_fnxxxxx:%s" % (feature_spec))
            serialized_tf_example = tf.placeholder(dtype=tf.string,
                                                   name='input_example_tensor')
            receiver_tensors = {'inputs': serialized_tf_example}
            features = tf.parse_example(serialized_tf_example, feature_spec)
            if not serving:
                return feature_spec
            return tf.estimator.export.ServingInputReceiver(
                features, receiver_tensors)

        def serving_input_receiver_fn_raw_tensor():
            # 0表示noexample格式
            tf.add_to_collection("IS_EXAMPLE", "0")
            user_columns, item_columns, user_group_features, item_group_features = feature_process()
            self.params['feature_columns'] = user_columns + item_columns
            features, receiver_tensors = xbd.get_receiver_tensors_from_dataschema_v2(
                self.params['feature_columns'],
                self.params['sample_data_schema'])
            if not serving:
                return receiver_tensors
            return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=receiver_tensors)

        if raw_feature_tensor:
            return serving_input_receiver_fn_raw_tensor
        else:
            return serving_input_receiver_fn

