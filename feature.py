# coding=utf-8
# time: 2022/11/17 4:12 下午
# author: shaojun7

import xembedding as xbd
import tensorflow as tf

MAX_CONSTANT = 9223372036854775807

def feature_process():

    user_columns = []
    item_columns = []

    # pickcat
    # wu211 用户自填性别
    u_wu211 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu211', vocabulary_list=['0','1','2'])
    user_columns.append(u_wu211)

    # wu215 用户挖掘年龄
    u_wu215 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu215', vocabulary_list=['00s','10s','90s','80s','70s','60s','50s','40s'])
    user_columns.append(u_wu215)

    # wu217 阅读者v类型
    u_wu217 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu217', vocabulary_list=['0','1','2','3','4','5','6','7','8','9','10'])
    user_columns.append(u_wu217)

    # wu2117 阅读者登录频次等级
    u_wu2117 = xbd.feature_column.PickcatsWithVocabularyList(key='u_wu2117', vocabulary_list=['0','1','2','3','4','5','6','7','8','9','10'])
    user_columns.append(u_wu2117)

    # wu2043 用户手机
    u_wu2043 = xbd.feature_column.CategoricalColumnWithHashBucket(key='u_wu2043', hash_bucket_size=50)
    user_columns.append(u_wu2043)


    # pickcats
    # wu2123 新标签体系用户三级标签长期兴趣
    u_wu2123 = xbd.feature_column.PickcatsWithHashBucket( key='u_wu2123',
                                                          inter_delimiter='|',
                                                          intra_delimiter='@',
                                                          std='0.01',
                                                          hash_bucket_size=100 * 100000)
    user_columns.append(u_wu2123)

    # wu21136 新标签体系用户二级标签长期兴趣
    u_wu21136 = xbd.feature_column.PickcatsWithHashBucket( key='u_wu21136',
                                                           inter_delimiter='|',
                                                           intra_delimiter='@',
                                                           std='0.01',
                                                           hash_bucket_size=5 * 100000)
    user_columns.append(u_wu21136)


    # ost107 超话标签
    r_ost107 = xbd.feature_column.PickcatsWithHashBucket(key='r_ost107',
                                                          inter_delimiter='|',
                                                          intra_delimiter='@',
                                                          std='0.01',
                                                          expired_min=60 * 24 * 2,
                                                          hash_bucket_size=10000)
    item_columns.append(r_ost107)

    # hash
    # ost108 超话属性
    r_ost108 = xbd.feature_column.CategoricalColumnWithHashBucket(key='r_ost108', hash_bucket_size=10 )
    item_columns.append(r_ost108)
    # wu2043 用户手机型号


    # ou101 uid自填省份
    u_ou101 = xbd.feature_column.CategoricalColumnWithHashBucket(key='u_ou101', hash_bucket_size=50)
    user_columns.append(u_ou101)

    # ou102 uid自填城市
    u_ou102 = xbd.feature_column.CategoricalColumnWithHashBucket(key='u_ou102', hash_bucket_size=2000)
    user_columns.append(u_ou102)

    # ost109 超话粉丝名
    r_ost109 = xbd.feature_column.CategoricalColumnWithHashBucket( key='r_ost109', hash_bucket_size=50 * 10000)
    item_columns.append(r_ost109)

    # 最近访问超话
    u_so1102 = xbd.feature_column.PickcatsWithHashBucket(
                                                        key="u_so1102",
                                                        inter_delimiter="|",
                                                        hash_bucket_size=5000000)
    user_columns.append(u_so1102)

    # 搜索词
    u_wur8100030 = xbd.feature_column.PickcatsWithHashBucket(
                                                        key="u_wur8100030",
                                                        inter_delimiter="|",
                                                        hash_bucket_size=50000000)
    user_columns.append(u_wur8100030)

    # 全站兴趣
    u_wu260850 = xbd.feature_column.PickcatsWithHashBucket(
                                                        key="u_wu260850",
                                                        inter_delimiter="|",
                                                        intra_delimiter="@",
                                                        hash_bucket_size=50000000)
    user_columns.append(u_wu260850)

    # 用户一天内互动过的超话ID序列(实时)
    u_wub52012 = xbd.feature_column.PickcatsWithHashBucket(
                                                        key="u_wub52012",
                                                        inter_delimiter="|",
                                                        hash_bucket_size=50000000)
    user_columns.append(u_wub52012)
    # 物料超话id
    r_stid = xbd.feature_column.CategoricalColumnWithHashBucket(key = "r_stid", hash_bucket_size=50000000)
    item_columns.append(r_stid)
    # 超话对应大v的uid
    r_ost116 = xbd.feature_column.CategoricalColumnWithHashBucket(key="r_ost116", hash_bucket_size=200000000)
    item_columns.append(r_ost116)
    # 超话L1类别
    r_ost129 = xbd.feature_column.PickcatsWithHashBucket(
                                                        key="r_ost129",
                                                        inter_delimiter="|",
                                                        intra_delimiter="@",
                                                        hash_bucket_size=1000000)
    item_columns.append(r_ost129)
    # 超话L2类别
    r_ost130 = xbd.feature_column.PickcatsWithHashBucket(
                                                        key="r_ost130",
                                                        inter_delimiter="|",
                                                        intra_delimiter="@",
                                                        hash_bucket_size=1000000)
    item_columns.append(r_ost130)


    # piesewise
    # ost101 超话帖子数
    r_ost101 = xbd.feature_column.BucketizedColumn(key='r_ost101', boundaries=[0.0, 100, 500, 1000, 2500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000 ])
    item_columns.append(r_ost101)
    # ost102 超话粉丝数
    r_ost102 = xbd.feature_column.BucketizedColumn(key='r_ost102', boundaries=[0.0, 100, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000 ])
    item_columns.append(r_ost102)
    # ost119 超话日增帖子数
    r_ost119 = xbd.feature_column.BucketizedColumn(key='r_ost119', boundaries=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 701.0, 801.0, 901.0, 1001.0, 2001.0, 3001.0, 4001.0, 5001.0, 6001.0, 7001.0, 8001.0, 9001.0, 10001.0, 20001.0, 30001.0, 40001.0, 50001.0, 60001.0, 70001.0, 80001.0, 90001.0, 100001.0, 200001.0, 300001.0, 400001.0, 500001.0, 600001.0, 700001.0, 800001.0, 900001.0, 1000001.0, 2000001.0, 3000001.0, 4000001.0, 5000000])
    item_columns.append(r_ost119)
    # ost121 超话日增粉丝数
    r_ost121 = xbd.feature_column.BucketizedColumn(key='r_ost121', boundaries=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 701.0, 801.0, 901.0, 1001.0, 2001.0, 3001.0, 4001.0, 5001.0, 6001.0, 7001.0, 8001.0, 9001.0, 10001.0, 20001.0, 30001.0, 40001.0, 50001.0, 60001.0, 70001.0, 80001.0, 90001.0, 100001.0, 200001.0, 300001.0, 400001.0, 500001.0, 600001.0, 700001.0, 800001.0, 900001.0, 1000000 ])
    item_columns.append(r_ost121)
    # 超话的人均刷新(7天)
    r_wst620006 = xbd.feature_column.BucketizedColumn(key='r_wst620006', boundaries=[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 500, 1000 ])
    item_columns.append(r_wst620006)
    # 超话的人均发博(7天)
    r_wst620007 = xbd.feature_column.BucketizedColumn(key='r_wst620007', boundaries=[0.0, 1.5, 2.0, 2.15, 2.3, 2.5, 2.65, 2.75, 2.9, 3.3, 3.5, 4.0, 4.5, 5.0, 6.5, 8.0, 9.0, 10, 50, 100 ])
    item_columns.append(r_wst620007)
    # 超话的次留率(7天)
    r_wst620008 = xbd.feature_column.BucketizedColumn(key='r_wst620008', boundaries=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87, 0.9, 1.0])
    item_columns.append(r_wst620008)
    # ost123 超话今日明星空降数
    r_ost123 = xbd.feature_column.BucketizedColumn(key='r_ost123', boundaries=[1.0, 3, 5, 8, 10, 15, 20, 30, 50, 70, 100, 130, 160, 200 ])
    item_columns.append(r_ost123)
    # ost124 超话最近3天明星空降数
    r_ost124 = xbd.feature_column.BucketizedColumn(key='r_ost124', boundaries=[0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 24, 39, 50, 100, 150, 200])
    item_columns.append(r_ost124)
    # ost125 超话最近7天明星空降数
    r_ost125 = xbd.feature_column.BucketizedColumn(key='r_ost125', boundaries=[0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300 ])
    item_columns.append(r_ost125)
    # ost126 超话阅读人数
    r_ost126 = xbd.feature_column.BucketizedColumn(key='r_ost126', boundaries=[0.0, 10, 50, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 500000, 1000000, 5000000, 10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000, 90000000, 100000000, 500000000, 1000000000 ])
    item_columns.append(r_ost126)
    # ost127 超话被提及数
    r_ost127 = xbd.feature_column.BucketizedColumn(key='r_ost127', boundaries=[0.0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 3000000, 5000000, 7000000, 9000000, 11000000, 13000000, 15000000, 17000000, 20000000, 50000000, 100000000])
    item_columns.append(r_ost127)
    # ost128 超话被提及人数
    r_ost128 = xbd.feature_column.BucketizedColumn(key='r_ost128', boundaries=[0.0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2500000, 5000000, 7500000, 10000000, 50000000])
    item_columns.append(r_ost128)
    # ost131 超话每小时活跃数
    r_ost131 = xbd.feature_column.BucketizedColumn(key='r_ost131', boundaries=[-1.0, 0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 5000, 10000 ])
    item_columns.append(r_ost131)
    # wu212 用户年龄
    u_wu212 = xbd.feature_column.BucketizedColumn(key='u_wu212', boundaries=[0.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 200])
    user_columns.append(u_wu212)

    user_group = [x for x in user_columns]
    user_group_features = xbd.feature_column.GroupColumn(key='user_group_features',
                                                    columns=user_group,
                                                    sparse=True,
                                                    use_conf=False)
    item_group = [x for x in item_columns]
    item_group_features = xbd.feature_column.GroupColumn(key='item_group_features',
                                                    columns=item_group,
                                                    sparse=True,
                                                    use_conf=False)

    return user_columns, item_columns, user_group_features, item_group_features


## useless
def get_feature_conf():
    columns = ['is_realread', 'is_addatten', 'is_click', 'ts', 'subscript', 'scene', 'uid', 'label', 'page_stid',
                   'r_stid', 'u_bf1001', 'u_ff1001', 'u_ost101', 'u_ost102', 'u_ost103', 'u_ost104', 'u_ost105',
                   'u_ost106', 'u_ost107', 'u_ost108', 'u_ost109', 'u_ost110', 'u_ost111', 'u_ost112', 'u_ost113',
                   'u_ost114', 'u_ost115', 'u_ost116', 'u_ost117', 'u_ost118', 'u_ost119', 'u_ost120', 'u_ost121',
                   'u_ost122', 'u_ost123', 'u_ost124', 'u_ost125', 'u_ost126', 'u_ost127', 'u_ost128', 'u_ost129',
                   'u_ost130', 'u_ost131', 'u_ost132', 'u_ou101', 'u_ou102', 'u_ou103', 'u_ou104', 'u_ou105', 'u_ou106',
                   'u_ou107', 'u_ou108', 'u_ou109', 'u_rsp101', 'u_rsp102', 'u_rsp103', 'u_rsp104', 'u_rsp105',
                   'u_rsp106', 'u_rsp107', 'u_rsp109', 'u_rsp111', 'u_rsp112', 'u_rsp113', 'u_rsp114', 'u_rsp115',
                   'u_rsp116', 'u_rsp117', 'u_rsp118', 'u_rsp119', 'u_rsp120', 'u_rsp121', 'u_rsp122', 'u_rsu101',
                   'u_rsu102', 'u_rsu103', 'u_rsu104', 'u_rsu105', 'u_rsu106', 'u_rsu107', 'u_sf1101', 'u_so1101',
                   'u_sth1000', 'u_wf4042', 'u_wu2043', 'u_wu211', 'u_wu21135', 'u_wu21136', 'u_wu2117', 'u_wu21173',
                   'u_wu21174', 'u_wu21175', 'u_wu21177', 'u_wu21178', 'u_wu21179', 'u_wu212', 'u_wu2123', 'u_wu21607',
                   'u_wu217', 'u_wu260714', 'u_wu260716', 'u_wu260717', 'u_wu260718', 'u_wub52012', 'r_itm101',
                   'r_ost101', 'r_ost102', 'r_ost103', 'r_ost104', 'r_ost105', 'r_ost106', 'r_ost107', 'r_ost108',
                   'r_ost109', 'r_ost110', 'r_ost111', 'r_ost112', 'r_ost113', 'r_ost114', 'r_ost115', 'r_ost116',
                   'r_ost117', 'r_ost118', 'r_ost119', 'r_ost120', 'r_ost121', 'r_ost122', 'r_ost123', 'r_ost124',
                   'r_ost125', 'r_ost126', 'r_ost127', 'r_ost128', 'r_ost129', 'r_ost130', 'r_ost131', 'r_ost132',
                   'r_rsp101', 'r_rsp102', 'r_rsp103', 'r_rsp104', 'r_rsp105', 'r_rsp106', 'r_rsp107', 'r_rsp109',
                   'r_rsp111', 'r_rsp112', 'r_rsp113', 'r_rsp114', 'r_rsp115', 'r_rsp116', 'r_rsp117', 'r_rsp118',
                   'r_rsp119', 'r_rsp120', 'r_rsp121', 'r_rsp122', 'r_rsu101', 'r_rsu102', 'r_rsu103', 'r_rsu104',
                   'r_rsu105', 'r_rsu106', 'r_rsu107', 'r_st_mid_pos', 'r_ut4616', 'r_ut4617', 'r_ut4618', 'r_ut4619',
                   'r_ut4620', 'r_ut4621', 'r_ut4622', 'r_ut4623', 'r_ut4624', 'r_ut4625', 'r_ut4626', 'r_ut4627',
                   'r_ut4628', 'r_ut4629', 'r_ut4630', 'r_ut4631', 'r_ut4632', 'r_ut4633', 'r_zst1000', 'r_zst2000',
                   'r_zst3000', 'bid', 'score', 'rc_srcs', 'r_wst620006', 'r_wst620007', 'r_wst620008', 'u_wur8100030',
                   'u_wur8100015', 'u_wu260850', 'u_so1102', 'u_wu215', 'r_genst50223', 'u_wu260803', 'c_containerid',
                   'ab', 'r_zm100', 'r_zm101', 'u_wu261161', 'u_wu261079', 'u_wu261080', 'u_wu260862']

    long_columns = []
    int_columns = []
    float_columns = []
    default_values = [[int(0)] if i in int_columns + long_columns else [0.0] if i in float_columns else ['0']
                      for i in columns]

    return columns, default_values
