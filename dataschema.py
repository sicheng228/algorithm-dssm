# coding=utf-8
import sys

sys.path.append("../..")
from xembedding.source.BaseSource import DataFormat
from xembedding.source.BaseSource import DataSchema


column_list = ['is_realread', 'is_addatten', 'is_click', 'ts', 'subscript', 'scene', 'uid', 'label', 'page_stid',
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

float_columns = ['is_addatten']


def get_data_schema():

    column_default = []

    float_columns = []

    for i in column_list:
        if i in float_columns:
            column_default.append(0.0)
        else:
            column_default.append("0")

    schema = DataSchema(column_list=column_list,
                        column_default=column_default,
                        data_format=DataFormat.CSV,
                        field_delim='\t',
                        na_value=["null", "NULL", "error_1", "exception", "|", "-1", "\\N", "", '', 'miss', 'None'])
    return schema


def get_prediction_schema():
    column_list = [
        'is_addatten'
    ]
    return column_list

def get_cols():
    types = []
    default_values = []
    for x in column_list:
        if x in float_columns:
            types.append('float')
            default_values.append(0.0)
        else:
            types.append('string')
            default_values.append('oov')
    return column_list, types, default_values

