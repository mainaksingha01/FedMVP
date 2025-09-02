def reset_cfg(cfg, args):
    cfg.EXP_NAME = args.exp_name
    ############# base-to-new generalization #############
    if args.exp_name == "cross_cls":
        cfg.DATASET.NAME_SPACE = ['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397']
        cfg.TRAIN.SPLIT = 'base'
        cfg.TEST.SPLIT = 'base&new'
        cfg.DATASET.TESTNAME_SPACE = ['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397']
    
    ############# cross dataset generalization #############
    elif args.exp_name == "cross_data":
        cfg.DATASET.NAME_SPACE = ["imagenet"]
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'ucf101', 'oxford_pets', 'food101', 'dtd', 'stanford_cars', 'sun397', 'eurosat']
    
    ############# singlesource multitarget domain generalization of ImageNet #############
    elif args.exp_name == "cross_domain":
        cfg.DATASET.NAME_SPACE = ["imagenet"]
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['imagenet-v2','imagenet-s','imagenet-a','imagenet-r','imagenet']
        
    ############# multisource singletarget domain generalization #############
    elif args.exp_name == "multisource_singletarget_pacs":
        cfg.DATASET.NAME_SPACE = ['pacs_artpainting', 'pacs_cartoon', 'pacs_photo']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['pacs_sketch']

    elif args.exp_name == "multisource_singletarget_office":
        cfg.DATASET.NAME_SPACE = ['officehome_art', 'officehome_clipart', 'officehome_product']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['officehome_realworld']
        
    elif args.exp_name == "multisource_singletarget_vlcs":
        cfg.DATASET.NAME_SPACE = ['vlcs_caltech', 'vlcs_labelme', 'vlcs_pascal']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['vlcs_sun']
    
    elif args.exp_name == "multisource_singletarget_terra":
        cfg.DATASET.NAME_SPACE = ['terra_incognita_l38', 'terra_incognita_l43', 'terra_incognita_l46']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['terra_incognita_l100']
        
    elif args.exp_name == "multisource_singletarget_domainnet":   
        cfg.DATASET.NAME_SPACE = ['domainnet_clipart', 'domainnet_infograph', 'domainnet_painting', 'domainnet_quickdraw', 'domainnet_real']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE =['domainnet_sketch']
  
    ############# singlesource multitarget domain generalization #############   
    elif args.exp_name == "singlesource_multitarget_pacs":
        cfg.DATASET.NAME_SPACE = ['pacs_artpainting']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['pacs_cartoon', 'pacs_photo', 'pacs_sketch']
        
    elif args.exp_name == "singlesource_multitarget_office":
        cfg.DATASET.NAME_SPACE = ['officehome_art']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['officehome_clipart', 'officehome_product', 'officehome_realworld']
    
    elif args.exp_name == "singlesource_multitarget_vlcs":   
        cfg.DATASET.NAME_SPACE = ['vlcs_caltech']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['vlcs_labelme', 'vlcs_pascal', 'vlcs_sun']
        
    elif args.exp_name == "singlesource_multitarget_terra":       
        cfg.DATASET.NAME_SPACE = ['terra_incognita_l38']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['terra_incognita_l43', 'terra_incognita_l46', 'terra_incognita_l100']
        
    elif args.exp_name == "singlesource_multitarget_domainnet":
        cfg.DATASET.NAME_SPACE = ['domainnet_clipart']
        cfg.TRAIN.SPLIT = 'all'
        cfg.TEST.SPLIT = 'all'
        cfg.DATASET.TESTNAME_SPACE = ['domainnet_quickdraw', 'domainnet_infograph', 'domainnet_painting', 'domainnet_real', 'domainnet_sketch']
    
    ###########################################################################################

    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.OUTPUT_DIR = args.output_dir
    cfg.RESUME = args.resume
    cfg.SEED = args.seed
    # cfg.TASK = args.task
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.OPTIM.MAX_EPOCH = args.num_epoch


    cfg.MODEL.D_CTX = args.depth_ctx
    cfg.MODEL.N_CTX = args.n_ctx
    cfg.MODEL.DEPTH = args.model_depth
    cfg.MODEL.NAME = args.model_name

    cfg.DATALOADER.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_CLASS_PER_CLIENT = args.num_cls_per_client
    cfg.TRAIN.AVAIL_PERCENT = args.avail_percent
    cfg.TRAIN.ALPHA = args.alpha