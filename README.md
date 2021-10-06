# self-supervised-learning
Zoo for self-supervised models

Parameters to set in yaml config file: 

model_name: 
    backbone: 
        model_backbone_name_1: 
            arch: string. architecture of the model
            pytorch_pretrained: bool. whether or not to load pretrained weights. 
            resume_path: string location of weights to load. can take value: 
                default - load from default pytorch path 
                path - path on machine to weights to load 
        model_backbone_name_2: 
            ...
    projection_head: 
        projection_head_name_1: 
            type: type of projection head 
            params: 
                ... add parameters param_name: setting