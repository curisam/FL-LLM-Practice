    """
    2025-08-21 17:40:06 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'val' has been reset and recreated. (sharded=True, world_size=4, rank=1, local_count=37, total=146)
    2025-08-21 17:40:06 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'val' has been reset and recreated. (sharded=True, world_size=4, rank=0, local_count=37, total=146)
    2025-08-21 17:40:06 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'val' has been reset and recreated. (sharded=True, world_size=4, rank=2, local_count=36, total=146)
    2025-08-21 17:40:06 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'val' has been reset and recreated. (sharded=True, world_size=4, rank=3, local_count=36, total=146)
    

    2025-08-21 17:40:06 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.
    2025-08-21 17:40:06 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.
    2025-08-21 17:40:06 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.
    2025-08-21 17:40:06 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    

    2025-08-21 17:40:09 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=2, local_total=36
    2025-08-21 17:40:09 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=1, local_total=37
    2025-08-21 17:40:09 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=3, local_total=36
    2025-08-21 17:40:09 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=0, local_total=37


    2025-08-21 17:40:09 (federatedscope.llm.trainer.reward_choice_trainer:543) INFO: [val|final] total=146, loss_sum=101.797127, avg_loss=0.697241, seen=146, correct=82, acc=0.561644
    
    
    2025-08-21 17:40:09 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.
    2025-08-21 17:40:09 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.
    2025-08-21 17:40:09 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.
    2025-08-21 17:40:09 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.


    2025-08-21 17:40:10 (federatedscope.core.workers.client:631) INFO: [DEBUG][after val] eval_metrics            = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:632) INFO: [DEBUG][after val] metrics (merged so far) = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:633) INFO: [DEBUG][after val] ctx.eval_metrics        = {}

    2025-08-21 17:40:10 (federatedscope.core.workers.client:631) INFO: [DEBUG][after val] eval_metrics            = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:632) INFO: [DEBUG][after val] metrics (merged so far) = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:633) INFO: [DEBUG][after val] ctx.eval_metrics        = {}

    2025-08-21 17:40:10 (federatedscope.core.workers.client:631) INFO: [DEBUG][after val] eval_metrics            = {'val_total': 146, 'val_loss': 101.79712677001953, 'val_avg_loss': 0.6972405943152022, 'val_seen': 146, 'val_correct': 82, 'val_acc': 0.5616438356164384}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:632) INFO: [DEBUG][after val] metrics (merged so far) = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:633) INFO: [DEBUG][after val] ctx.eval_metrics        = {'val_total': 146, 'val_loss': 101.79712677001953, 'val_avg_loss': 0.6972405943152022, 'val_seen': 146, 'val_correct': 82, 'val_acc': 0.5616438356164384}

    2025-08-21 17:40:10 (federatedscope.core.workers.client:631) INFO: [DEBUG][after val] eval_metrics            = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:632) INFO: [DEBUG][after val] metrics (merged so far) = {}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:633) INFO: [DEBUG][after val] ctx.eval_metrics        = {}



    2025-08-21 17:40:10 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'val', 'Rank': '3/4', 'Local': True, 'Results': {'val_total': 36, 'val_loss': 25.305973172187805, 'val_avg_loss': 0.702943699227439, 'val_seen': 36, 'val_correct': 19, 'val_acc': 0.5277777777777778}}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'val', 'Rank': '1/4', 'Local': True, 'Results': {'val_total': 37, 'val_loss': 27.17430353164673, 'val_avg_loss': 0.7344406359904522, 'val_seen': 37, 'val_correct': 17, 'val_acc': 0.4594594594594595}}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'val', 'Rank': '0/4', 'Local': True, 'Results': {'val_total': 37, 'val_loss': 24.979133307933807, 'val_avg_loss': 0.675111711025238, 'val_seen': 37, 'val_correct': 23, 'val_acc': 0.6216216216216216}}
    2025-08-21 17:40:10 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'val', 'Rank': '2/4', 'Local': True, 'Results': {'val_total': 36, 'val_loss': 24.33772349357605, 'val_avg_loss': 0.6760478748215569, 'val_seen': 36, 'val_correct': 23, 'val_acc': 0.6388888888888888}}


    2025-08-21 17:40:10 (federatedscope.core.workers.client:382) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'val', 'Aggregated': True, 'Results_raw': {'val_total': 146, 'val_loss': 101.79712677001953, 'val_avg_loss': 0.6972405943152022, 'val_seen': 146, 'val_correct': 82, 'val_acc': 0.5616438356164384}}


    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.

    2025-08-21 17:40:10 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'test' has been reset and recreated. (sharded=True, world_size=4, rank=1, local_count=10, total=40)
    2025-08-21 17:40:10 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'test' has been reset and recreated. (sharded=True, world_size=4, rank=0, local_count=10, total=40)
    2025-08-21 17:40:10 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'test' has been reset and recreated. (sharded=True, world_size=4, rank=3, local_count=10, total=40)
    2025-08-21 17:40:10 (federatedscope.llm.trainer.reward_choice_trainer:217) INFO: Dataloader for 'test' has been reset and recreated. (sharded=True, world_size=4, rank=2, local_count=10, total=40)


    2025-08-21 17:40:10 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.
    2025-08-21 17:40:10 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.
    2025-08-21 17:40:10 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.
    2025-08-21 17:40:10 (federatedscope.llm.trainer.trainer:345) INFO: Re-creating Accelerator for the new round.

    2025-08-21 17:40:11 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=1, local_total=10
    2025-08-21 17:40:11 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=2, local_total=10
    2025-08-21 17:40:11 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=0, local_total=10
    2025-08-21 17:40:11 (federatedscope.core.trainers.trainer:425) INFO: [agg debug] using_accel=True, world=4, rank=3, local_total=10

    2025-08-21 17:40:11 (federatedscope.llm.trainer.reward_choice_trainer:543) INFO: [test|final] total=40, loss_sum=28.575809, avg_loss=0.714395, seen=40, correct=19, acc=0.475000

    2025-08-21 17:40:11 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.
    2025-08-21 17:40:11 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.
    2025-08-21 17:40:11 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.
    2025-08-21 17:40:11 (federatedscope.llm.trainer.trainer:635) INFO: Accelerator object has been deleted.

    2025-08-21 17:40:12 (federatedscope.core.workers.client:631) INFO: [DEBUG][after test] eval_metrics            = {'test_total': 40, 'test_loss': 28.575809478759766, 'test_avg_loss': 0.7143952369689941, 'test_seen': 40, 'test_correct': 19, 'test_acc': 0.475}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:632) INFO: [DEBUG][after test] metrics (merged so far) = {'val_total': 146, 'val_loss': 101.79712677001953, 'val_avg_loss': 0.6972405943152022, 'val_seen': 146, 'val_correct': 82, 'val_acc': 0.5616438356164384}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:633) INFO: [DEBUG][after test] ctx.eval_metrics        = {'test_total': 40, 'test_loss': 28.575809478759766, 'test_avg_loss': 0.7143952369689941, 'test_seen': 40, 'test_correct': 19, 'test_acc': 0.475}

    2025-08-21 17:40:12 (federatedscope.core.workers.client:631) INFO: [DEBUG][after test] eval_metrics            = {}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:632) INFO: [DEBUG][after test] metrics (merged so far) = {}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:633) INFO: [DEBUG][after test] ctx.eval_metrics        = {}


    2025-08-21 17:40:12 (federatedscope.core.workers.client:631) INFO: [DEBUG][after test] eval_metrics            = {}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:632) INFO: [DEBUG][after test] metrics (merged so far) = {}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:633) INFO: [DEBUG][after test] ctx.eval_metrics        = {}

    2025-08-21 17:40:12 (federatedscope.core.workers.client:631) INFO: [DEBUG][after test] eval_metrics            = {}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:632) INFO: [DEBUG][after test] metrics (merged so far) = {}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:633) INFO: [DEBUG][after test] ctx.eval_metrics        = {}


    2025-08-21 17:40:12 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'test', 'Rank': '0/4', 'Local': True, 'Results': {'test_total': 10, 'test_loss': 7.167285442352295, 'test_avg_loss': 0.7167285442352295, 'test_seen': 10, 'test_correct': 4, 'test_acc': 0.4}}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'test', 'Rank': '1/4', 'Local': True, 'Results': {'test_total': 10, 'test_loss': 6.835325241088867, 'test_avg_loss': 0.6835325241088868, 'test_seen': 10, 'test_correct': 5, 'test_acc': 0.5}}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'test', 'Rank': '3/4', 'Local': True, 'Results': {'test_total': 10, 'test_loss': 7.136165022850037, 'test_avg_loss': 0.7136165022850036, 'test_seen': 10, 'test_correct': 5, 'test_acc': 0.5}}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:369) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'test', 'Rank': '2/4', 'Local': True, 'Results': {'test_total': 10, 'test_loss': 7.437033653259277, 'test_avg_loss': 0.7437033653259277, 'test_seen': 10, 'test_correct': 5, 'test_acc': 0.5}}


    2025-08-21 17:40:12 (federatedscope.core.workers.client:382) INFO: {'Role': 'Client #1', 'Round': 1, 'Split': 'test', 'Aggregated': True, 'Results_raw': {'test_total': 40, 'test_loss': 28.575809478759766, 'test_avg_loss': 0.7143952369689941, 'test_seen': 40, 'test_correct': 19, 'test_acc': 0.475}}


    2025-08-21 17:40:12 (federatedscope.core.workers.client:668) INFO: [DEBUG][before write] agg_all={'test_total': 40, 'test_loss': 28.575809478759766, 'test_avg_loss': 0.7143952369689941, 'test_seen': 40, 'test_correct': 19, 'test_acc': 0.475}, metrics={'val_total': 146, 'val_loss': 101.79712677001953, 'val_avg_loss': 0.6972405943152022, 'val_seen': 146, 'val_correct': 82, 'val_acc': 0.5616438356164384, 'test_total': 40, 'test_loss': 28.575809478759766, 'test_avg_loss': 0.7143952369689941, 'test_seen': 40, 'test_correct': 19, 'test_acc': 0.475}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:668) INFO: [DEBUG][before write] agg_all={}, metrics={}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:668) INFO: [DEBUG][before write] agg_all={}, metrics={}
    2025-08-21 17:40:12 (federatedscope.core.workers.client:668) INFO: [DEBUG][before write] agg_all={}, metrics={}



    2025-08-21 17:40:12 (federatedscope.core.workers.client:678) INFO: [DEBUG] combined keys=['test_total', 'test_loss', 'test_avg_loss', 'test_seen', 'test_correct', 'test_acc', 'val_total', 'val_loss', 'val_avg_loss', 'val_seen', 'val_correct', 'val_acc'], has_val=True, has_test=True
    2025-08-21 17:40:12 (federatedscope.core.workers.client:678) INFO: [DEBUG] combined keys=[], has_val=False, has_test=False
    2025-08-21 17:40:12 (federatedscope.core.workers.client:678) INFO: [DEBUG] combined keys=[], has_val=False, has_test=False
    2025-08-21 17:40:12 (federatedscope.core.workers.client:678) INFO: [DEBUG] combined keys=[], has_val=False, has_test=False


    2025-08-21 17:40:12 (root:790) INFO: Find new best result: {'client #1': {'test_loss': 28.575809478759766, 'test_total': 40, 'test_avg_loss': 0.7143952369689941, 'test_seen': 40, 'test_correct': 19, 'test_acc': 0.475, 'val_total': 146, 'val_loss': 101.79712677001953, 'val_avg_loss': 0.6972405943152022, 'val_seen': 146, 'val_correct': 82, 'val_acc': 0.5616438356164384}}
    
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
 
    """