import os
import copy

opt = dict()

opt['device'] = 'cuda:0'
opt['seed'] = '2021'
opt['data'] = 'pubmed'
opt['epochs'] = 600
opt['lr'] = 1e-3
opt['input_dim'] = 500
opt['momentum'] = 0.9
opt['alpha'] = 0.05
opt['beta'] = 0.7
opt['drop_edge'] = 0.4
opt['drop_feat1'] = 0.2
opt['drop_feat2'] = 0.2


def command(opt):
    script = 'python train.py'
    for opt, val in opt.items():
        script += ' --' + opt + ' ' + str(val)
    return script


def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(command(opt_))


if __name__ == '__main__':
    run(opt)