# self-supervised-learning
Parameters to set in yaml config file for byol: 

```
byol: 
    backbone: 
        encoder: 
            arch: string. architecture of the model
            pytorch_pretrained: bool. whether or not to load pretrained weights. 
            resume_path: string location of weights to load. can take value: 
                default - load from default pytorch path 
                path - path on machine to weights to load 
    projection_head: 
        projection: 
            type: type of projection head 
            params: 
                ... add parameters param_name: setting
```