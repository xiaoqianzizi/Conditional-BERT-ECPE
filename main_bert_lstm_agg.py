# encoding: utf-8
# @author: cq
# reference: zxding, xinhchen
# email: 483412504@qq.com


import numpy as np
import tensorflow as tf
import transformer as trans
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.tf_funcs import *
from utils.prepare_data import *
from bert import modeling, optimization, tokenization

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', './nega_data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per document')
tf.app.flags.DEFINE_integer('max_cau_num', 3, 'max number of causes per document')
tf.app.flags.DEFINE_integer('max_sen_len_bert', 60, 'max number of tokens per sentence for Bert')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 50, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_dir', './log', 'directory path of log file')
tf.app.flags.DEFINE_integer('max_to_keep', 5, 'maximum number of checkpoints')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 30, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
tf.app.flags.DEFINE_string('bert_base_dir', './BERTmodel/chinese_L-12_H-768_A-12/', 'base dir of pretrained bert')
tf.app.flags.DEFINE_float('bert_hidden_kb', 0.9, 'keep prob for bert')
tf.app.flags.DEFINE_float('bert_attention_kb', 0.8, 'keep prob for bert')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 3, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 4e-6, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.8, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-6, 'l2 regularization')
tf.app.flags.DEFINE_integer('num_heads', 2, 'number of heads of self attention')

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(x_bert, sen_len_bert, doc_len, is_training, keep_prob1, keep_prob2, y, emo, cau, con):

    def get_bert_s(x_bert, sen_len_bert, is_training, feature_mask):
        x_bert = tf.reshape(x_bert, [-1, FLAGS.max_sen_len_bert])
        sen_len_bert = tf.reshape(sen_len_bert, [-1])
        x_bert_mask = tf.cast(tf.sequence_mask(sen_len_bert, FLAGS.max_sen_len_bert), tf.int32)
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_base_dir + 'bert_config.json')
        bert_config.hidden_dropout_prob, bert_config.attention_probs_dropout_prob = 1 - FLAGS.bert_hidden_kb, 1 - FLAGS.bert_attention_kb
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=x_bert,
            input_mask=x_bert_mask)
        s = model.get_pooled_output()
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, s.shape[-1].value])
        s = s * feature_mask
        s = tf.layers.dense(s, 4*FLAGS.n_hidden, use_bias=True)
        return s


    feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
    s = get_bert_s(x_bert, sen_len_bert, is_training, feature_mask)
    # [-1, 1, max_doc_len] * [-1, max_doc_len, 2*hidden] = [-1, 1, 2*hidden_size]
    s_emo = tf.matmul(emo, s)
    s_cau = tf.matmul(cau, s)
    s_woc = tf.concat([s_emo, s_cau], 1)
    s_woc = tf.reshape(s_woc, [-1, 4, 4*FLAGS.n_hidden])

    s_sen = biLSTM(s, doc_len, n_hidden=2*FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_lstm_layer')
    s_emo_wc = tf.matmul(emo, s_sen)
    s_cau_wc = tf.matmul(cau, s)
    s_wc = tf.concat([s_emo_wc, s_cau_wc], 1)
    s_wc = tf.reshape(s_wc, [-1, 4, 4*FLAGS.n_hidden])

    loss_wc = tf.constant(0, tf.float32)
    reg_wc = tf.constant(0, tf.float32)
    reg_woc = tf.constant(0, tf.float32)

    with tf.name_scope('context_prediction'):
        # PAIR based prediction
        pred_woc, reg_woc = softmax_part(s_woc, 8*2*FLAGS.n_hidden, keep_prob2, FLAGS.n_class, 'softmax_w_woc', 'softmax_b_woc')

        # CONTEXT based prediction
        pred_wc, reg_wc = softmax_part(s_wc, 8*2*FLAGS.n_hidden, keep_prob2, FLAGS.n_class, 'softmax_w_wc', 'softmax_b_wc')

        # AGGREGATE PAIR and CONTEXT based prediction based on attention
        # calculate the difference between pred_woc and [0,1] and treat it as weight
        weight = tf.reshape(pred_woc[:, 1], [-1, 1])
        pred_final = weight * pred_woc + (1-weight) * pred_wc
        pred_final /= tf.reshape(tf.reduce_sum(pred_final, 1), [-1, 1])

        loss_wc = -tf.reduce_sum(y * tf.log(pred_final)) / tf.cast(tf.shape(x_bert)[0], tf.float32)

    loss = loss_wc
    reg = reg_wc + reg_woc

    return loss, pred_final, reg


def softmax_part(sent_e, n_feature, keep_prob, o_feature, w_name, b_name):
    s1 = tf.reshape(sent_e, [-1, n_feature])
    s1 = tf.nn.dropout(s1, keep_prob=keep_prob)
    w = get_weight_varible(name=w_name, shape=[n_feature, o_feature])
    b = get_weight_varible(name=b_name, shape=[o_feature])
    before_softmax = tf.matmul(s1, w) + b
    before_softmax = tf.reshape(before_softmax, [-1, o_feature])
    pred = tf.nn.softmax(before_softmax)
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, sen_len, doc_len, is_training, keep_prob1, keep_prob2, y, batch_size, emo, cau, con, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], is_training, keep_prob1, keep_prob2, y[index], emo[index], cau[index], con[index]]
        yield feed_list, len(index)


def run():
    logger = get_logger(FLAGS.log_file_dir, FLAGS.scope)

    res_dir = './result/' + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')) + FLAGS.scope
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    print('build model...')

    # if only emotion and cause, then should only be 4 instead of FLAGS.max_doc_len
    x_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len_bert])
    sen_len_bert = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.float32)
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
    emo = tf.placeholder(tf.float32, [None, 1, FLAGS.max_doc_len])
    cau = tf.placeholder(tf.float32, [None, FLAGS.max_cau_num, FLAGS.max_doc_len])
    con = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len])
    placeholders = [x_bert, sen_len_bert, doc_len, is_training, keep_prob1, keep_prob2, y, emo, cau, con]

    loss, pred_wc, reg = build_model(x_bert, sen_len_bert, doc_len, is_training, keep_prob1, keep_prob2, y, emo, cau, con)
    loss_op = loss + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    true_y_op = y

    print('build model done!\n')

    def init_from_bert_checkpoint():
        init_checkpoint = FLAGS.bert_base_dir + 'bert_model.ckpt'
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for idx, var in enumerate(tvars):
            init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variable_names else ""
            tf.logging.info("var-index %s:  name = %s, shape = %s%s", idx, var.name, var.shape, init_string)

    def get_bert_tokenizer():
        do_lower_case = True
        tokenization.validate_case_matches_checkpoint(do_lower_case, FLAGS.bert_base_dir + 'bert_model.ckpt')
        return tokenization.FullTokenizer(vocab_file=FLAGS.bert_base_dir + 'vocab.txt', do_lower_case=do_lower_case)



    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        acc_list, p_list, r_list, f1_list, auc_list = [], [], [], [], []
        tokenizer = get_bert_tokenizer()

        for fold in range(1, 11):
            init_from_bert_checkpoint()
            sess.run(tf.global_variables_initializer())
            # train for one fold
            logger.info('############# fold {} begin ###############'.format(fold))

            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold)
            test_file_name = 'fold{}_test.txt'.format(fold)
            _, tr_x, tr_sen_len, tr_doc_len, tr_y, tr_emo, tr_cau, tr_con, _ = load_data('./nega_data/'+train_file_name, tokenizer, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_cau_num)
            te_doc_id, te_x, te_sen_len, te_doc_len, te_y, te_emo, te_cau, te_con, _ = load_data('./nega_data/'+test_file_name, tokenizer, FLAGS.max_doc_len, FLAGS.max_sen_len_bert, FLAGS.max_cau_num)

            max_p = max_r = max_acc = max_f1 = max_auc = -1.
            logger.info('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))

            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, 1, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, FLAGS.batch_size, tr_emo, tr_cau, tr_con):
                    _, loss, pred_y, true_y = sess.run(
                        [optimizer, loss_op, pred_wc, true_y_op], feed_dict=dict(zip(placeholders, train)))
                    #if step % 1 == 0:
                    #logger.info('step {}: train loss {:.4f} '.format(step, loss))
                    #acc, p, r, f1, auc = acc_prf(pred_y, true_y)
                    #logger.info('predict: train acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f} auc {:.4f}'.format(acc, p, r, f1, auc))

                    #step = step + 1

                # test
                pred_test = []
                true_test = []
                for test, _ in get_batch_data(te_x, te_sen_len, te_doc_len, 0, 1., 1., te_y, FLAGS.batch_size, te_emo, te_cau, te_con, test=True):
                    loss, pred_y, true_y = sess.run(
                        [loss_op, pred_wc, true_y_op], feed_dict=dict(zip(placeholders, test)))
                    #logger.info('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time()-start_time))
                    for g in range(len(pred_y)):
                        pred_test.append(pred_y[g])
                        true_test.append(true_y[g])

                acc, p, r, f1, auc = acc_prf(np.array(pred_test), np.array(true_test))
                logger.info('prediction: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f} auc {:.4f}'.format(acc, p, r, f1, auc))

                if f1 > max_f1 and r < 0.85:
                    max_acc, max_p, max_r, max_f1, max_auc = acc, p, r, f1, auc
                    logger.info('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f} max_auc {:.4f}\n'.format(max_acc, max_p, max_r, max_f1, max_auc))
                    logger.info('Best result updated in Iteration {}'.format(i))
                    tmp_file = open(res_dir + "/best_pred_for_fold_{}.txt".format(fold), 'w')
                    for g in range(len(te_doc_id)):
                        tmp_file.write(str(te_doc_id[g]))
                        tmp_file.write('\t')
                        tmp_file.write(str(true_test[g]))
                        tmp_file.write('\t')
                        tmp_file.write(str(pred_test[g]))
                        tmp_file.write('\n')
                    tmp_file.close()

            logger.info('Optimization Finished!\n')
            logger.info('############# fold {} end ###############'.format(fold))
            acc_list.append(max_acc)
            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)
            auc_list.append(max_auc)

        print_training_info()
        all_results = [acc_list, p_list, r_list, f1_list, auc_list]
        acc_avg, p_avg, r_avg, f1_avg, auc_avg = map(lambda x: np.array(x).mean(), all_results)
        logger.info('\nPrediction: test auc in 10 fold: {}'.format(np.array(f1_list).reshape(-1, 1)))
        logger.info('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f} auc {:.4f}\n'.format(acc_avg, p_avg, r_avg, f1_avg, auc_avg))
        print_time()


def main(_):
    FLAGS.scope = 'bert_lstm_agg'
    run()


if __name__ == '__main__':
    tf.app.run()
