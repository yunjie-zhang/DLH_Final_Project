import sys
from utils.configs import cfg
from utils.record_log import RecordLog
import numpy as np
from model_bitenet import BiteNet as Model
import os
from dataset.dataset_full import VisitDataset
import warnings
import heapq
import operator
import tensorflow as tf
from utils.evaluation import ConceptEvaluation as CodeEval, \
    EvaluationTemplate as Evaluation
warnings.filterwarnings('ignore')
logging = RecordLog()


def train():

    visit_threshold = cfg.visit_threshold
    epochs = cfg.max_epoch
    batch_size = cfg.train_batch_size

    data_set = VisitDataset()
    data_set.prepare_data(visit_threshold)
    data_set.build_dictionary()
    data_set.load_data()
    code_eval = CodeEval(data_set, logging)
    print(data_set.train_context_codes.shape)
    print(data_set.train_intervals.shape)
    print(data_set.train_labels_2.shape)

    model = Model(data_set)
    model.build_network()
    model.model.fit(x=[data_set.train_context_codes,data_set.train_intervals],
                    y=data_set.train_labels_1,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=([data_set.dev_context_codes,data_set.dev_intervals], data_set.dev_labels_1)
                    # , callbacks=[es]
                    )

    metrics = model.model.evaluate([data_set.test_context_codes, data_set.test_intervals], data_set.test_labels_1)
    print(metrics)
    log_str = 'Single fold accuracy is {}'.format(metrics[1])
    logging.add(log_str)

    trues = data_set.test_labels_1
    predicts = model.model.predict([data_set.test_context_codes, data_set.test_intervals])
    np.save('dx_true_labels', data_set.test_labels_1)
    np.save('dx_predict_labels', predicts)

    preVecs = []
    trueVecs = []
    for i in range(predicts.shape[0]):
        preVec = [rk[0] for rk in heapq.nlargest(30, enumerate(predicts[i]), key=operator.itemgetter(1))]
        preVecs.append(preVec)
        trueVec = [rk[0] for rk in
                   heapq.nlargest(np.count_nonzero(trues[i]), enumerate(trues[i]), key=operator.itemgetter(1))]
        trueVecs.append(trueVec)
    recalls = Evaluation.recall_top(trueVecs, preVecs)
    logging.add("Precision@k")
    logging.add(recalls[0])

    logging.done()


def test():
    pass


def main():
    if cfg.train:
        train()
    else:
        test()


def output_model_params():
    logging.add()
    logging.add('==>model_title: ' + cfg.model_name[1:])
    logging.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            logging.add('%s: %s' % (key, value))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    main()