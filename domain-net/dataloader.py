import utils


class Trainer(object):
    def __init__(self,args):
        super(Trainer,self).__init__()
        self.args = args
        self.setup_loader()

    def setup_loader(self):
        self.trg_train_listloader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT, \
            os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.TARGET + '_train.txt'))
        
        ################################### Source Target domain data loader  ###################################
        self.src_image_set = data_loader.load_src_trainset()
        self.trg_paths, _ = utils.loadlines(os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.TARGET + '_train.txt'))
        probs, plabels = utils.load_trg_plabels()
        trg_paths, self.trg_probs, self.trg_varlabels = utils.filter_trg_plabels(self.trg_paths, probs, plabels) 
         
        if cfg.DATA_LOADER.ITER == -1:
            if cfg.MODEL.SOURCE_ONLY == True:
                src_paths, _ = utils.loadlines(os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.SOURCE + '_train.txt'))
                cfg.DATA_LOADER.ITER = len(src_paths) // cfg.TRAIN.BATCH_SIZE
            else:
                cfg.DATA_LOADER.ITER = len(trg_paths) // cfg.TRAIN.BATCH_SIZE

        cls_info, self.src_train_loader = data_loader.load_mergesrc_train(self.distributed, self.src_image_set)
        self.trg_train_loader = data_loader.load_trg_train(self.distributed, trg_paths, self.trg_varlabels, cls_info)
        ########################################################################################################

        ########################################### load test loader ###########################################
        # Target test loader
        self.trg_test_loader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT, \
            os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', cfg.DATA_LOADER.TARGET + '_test.txt'))

        self.trg_real_loader = data_loader.load_test(cfg.DATA_LOADER.DATA_ROOT, \
            os.path.join(cfg.DATA_LOADER.DATA_ROOT, 'list', 'real_test.txt'))
        ########################################################################################################

