# -*- coding: utf-8 -*-

import cfg
import pandas as pd
import numpy as np
from utility import apply_border


p1 = np.loadtxt(cfg.path_processed + 'mikhail_model1.txt')
p2 = np.loadtxt(cfg.path_processed + 'mikhail_model2.txt')
p3 = np.loadtxt(cfg.path_processed + 'dmitry_model1.txt')
p4 = np.loadtxt(cfg.path_processed + 'dmitry_model2.txt')

ens = apply_border(p1+p2+p3+p4,[0.33, 0.6, 0.77])

idx = pd.read_csv(cfg.path_test).id.values.astype(int)
#
submission = pd.DataFrame({"id": idx, "prediction": ens.astype(np.int32)})
submission.to_csv(cfg.path_submit + "final_solution.csv", index=False)

