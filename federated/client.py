from model.FedMVP import FedMVP
from dataloader.dm_federated import TrainDataManager
from federated.utils import *
import torch.nn.functional as F
from federated.base_trainer import TrainerBase
import os

class Client(TrainerBase):
    """A local client with frozen clip and FL meta_net and private training data"""
    def __init__(self, cfg, client_id,dataname,available_cls,clip_model):
        super().__init__()
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.client_id = client_id

        self.cfg = cfg
        self.build_data_loader(dataname,available_cls)
        self.build_model(clip_model)


    def build_data_loader(self,dataname,available_cls):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = TrainDataManager(self.cfg, dataname,available_cls)

        self.train_loader = dm.train_loader
        # self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.available_classes = dm.available_classes
        self.data_name = dm.data_name

    def build_model(self,clip_model):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.model_name = cfg.MODEL.NAME
        
        self.model = FedMVP(cfg, clip_model,device = self.device)

        #print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                
        for param in self.model.prompt_learner.crossattn.parameters():
            param.requires_grad = True 

        for param in [self.model.prompt_learner.cross_loraparams]:
            param.requires_grad = False
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            #self.model = nn.DataParallel(self.model)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

    def train(self, num_round, early_stop_threshold=0.5):
        self.set_model_mode("train")
        losses = MetricMeter()

        dataname = self.data_name
        classnames = self.available_classes

        # Track if LoRA is activated
        if not self.model.prompt_learner.use_lora:  
            print(f"Training without LoRA fine-tuning for client {self.client_id}")

        # Store previous loss to detect stabilization
        prev_loss = float('inf')

        for batch in self.train_loader:
            loss, acc = self.forward_backward(batch, dataname, classnames)

        # Check if loss is below threshold and LoRA isn't already enabled
        if loss.item() < early_stop_threshold and not self.model.prompt_learner.use_lora:
            print(f"Early stopping condition met! Switching to LoRA fine-tuning at client {self.client_id}")

            # Freeze `crossattn_vis`
            for param in self.model.prompt_learner.crossattn.parameters():
                param.requires_grad = False  
            # Enable LoRA fine-tuning
            for param in self.model.prompt_learner.cross_loraparams.parameters():
                param.requires_grad = True              
            self.model.prompt_learner.use_lora = True
            print("use_lora set to True")
            
            lora_params = 0
            for param in [self.model.prompt_learner.cross_loraparams]:
                if hasattr(param, 'parameters'):
                    lora_params += sum(p.numel() for p in param.parameters() if p.requires_grad)

            print(f"Number of LoRA Trainable Parameters: {lora_params}")

        self.model_backward_and_update(loss)
        prev_loss = loss.item()  

        # Log training info
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        losses.update(loss_summary)

        info = [
            f"epoch [{num_round + 1}/{self.max_epoch}]",
            f"client_id [{self.client_id}]",
            f"{dataname}",
            f"{losses}",
            f"lr {self.get_current_lr():.4e}",
        ]
        print(" ".join(info))

        self.update_lr()
        local_updates = self.model.prompt_learner.state_dict()
        return local_updates

    def load_meta(self, global_net):
        self.model.prompt_learner.load_state_dict(global_net)


    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def forward_backward(self, batch, dataname, classnames):
        images, labels, cnames = self.parse_batch(batch)

        image_output, vis_score = self.model(images, classnames, dataname)
        loss_ori = F.cross_entropy(image_output, labels)
        loss = loss_ori + self.cfg.TRAIN.ALPHA*vis_score
        return loss,compute_accuracy(image_output, labels)[0].item()

    def parse_batch(self, batch):
        input = batch["img"]
        label = batch["label"]
        cname = batch["cname"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, cname

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]
    def model_inference(self, input, classnames, dataname):
        return self.model(input, classnames, dataname)[0]