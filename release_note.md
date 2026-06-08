## VeRL-Omni 0.1 Release Note

- Model
    - QwenImage
    - Wan2.2
    - Bagel
        - [ ] 120 step, model colapse, can do early stop  @mike
    - Qwen3Omni-Thinker
        - [ ] FSDP Trainer, too many patches, code to be refactor   @qingan (社区主开发验证, Rice Univiersity) @didan (内部验证重构)
        - [ ] megatron trainer, PR to be merged  @wei (社区主开发验证, 滴滴AI Infra）

- Algorithm
    - DPO (for QwenImage & SD3.5)
    - FlowGRPO
    - MixGRPO
    - DiffusionNFT

- Features
    - [ ] Async reward compute, add design doc  @andy
    - Rollout correction
    - [ ] Step-wise diffusion continuous batching, e2e training convergence @Long
    - [ ] Prompt embedding cache, e2e training convergence  @Long

- Scalability
    - [ ] Multi-node support - 2 nodes  @Long @hyx
    - [ ] 8 or 16 nodes

- CI & UX
    - [ ] prepare dockers, image can put to Quay
        - [ ] GPU
    - [ ] standard convergence test  @Andy

- Upstream version rebase
    - vLLM-Omni 0.22
        - [ ] Fix model collapse, FA3 alignement.  @hyx
    - verl v0.8

- Performance Benchmarking
    - [ ] QwenImage with all optimized features (async reward, rollout correction, diffusion CB, prompt embedding cache), single-node

- [ ] NPU support for the about tasks

- 宣传相关
    - [ ] ICML海报    @didan
    - [ ] 开源社区介绍ppt (青稞AI邀请)
    - [ ] 公司内部介绍ppt 
    - [ ] v0.1版本宣传计划

