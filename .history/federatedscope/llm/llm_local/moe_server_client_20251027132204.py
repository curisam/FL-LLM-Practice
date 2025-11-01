"""
Server and client implementations for soft‑assignment MoE training.

This module provides two variants of the GFL algorithm described in
the FedMoE design.  Instead of hard assigning each client to a single
adapter (as in FedBiscuit), we compute a soft weight vector
``w_{u,m}`` for each client ``m`` and expert ``u`` at the end of the
E‑step (evaluation step).  These weights are updated using an
exponential moving average (EMA) of per‑adapter validation accuracies
or rewards.  During the M‑step, clients either train all experts
simultaneously (Full‑MoE) or train a single fused default adapter
(Fusion‑MoE).  Servers aggregate updates using client‑specific
weights.

Classes
-------
FullMoEServer
    Extends ``LLMMultiLoRAServer`` to implement soft assignment and
    weight dispatch for Full‑MoE.

FusionMoEServer
    Extends ``LLMMultiLoRAServer`` to implement soft assignment and
    weight dispatch for Fusion‑MoE.

FullMoEClient
    Extends ``LLMMultiLoRAClient`` to support receipt of weight
    vectors and weighted training of all adapters.

FusionMoEClient
    Extends ``LLMMultiLoRAClient`` to support receipt of weight
    vectors and fused default‑adapter training.

Usage
-----
Register these classes in your YAML configuration using the
``server.type`` and ``client.type`` keys.  For example:

.. code-block:: yaml

   # Full‑MoE
   federate:
     server_type: llmfullmoeserver
     client_type: llmfullmoeclient
   llm:
     trainer:
       type: llmfullmoetrainer
     aggregator:
       type: llmfullmoeaggregator

   # Fusion‑MoE
   federate:
     server_type: llmfusionmoeserver
     client_type: llmfusionmoeclient
   llm:
     trainer:
       type: llmfusionmoetrainer
     aggregator:
       type: llmfusionmoeaggregator

You may also set ``llm.adapter.ema_beta`` (default 0.9) to control
the weight update speed.
"""

import logging
import math
from typing import Dict, List, Optional, Any

from federatedscope.core.message import Message
from federatedscope.llm.llm_local.server import LLMMultiLoRAServer
from federatedscope.llm.llm_local.client import LLMMultiLoRAClient

logger = logging.getLogger(__name__)


class _BaseMoEServer(LLMMultiLoRAServer):
    """Base server implementing soft assignment of clients to experts.

    This class overrides the grouping callbacks of
    ``LLMMultiLoRAServer``.  Instead of assigning each client to a
    single expert, the server collects per‑adapter accuracies from
    clients and updates client‑specific weight vectors ``w_vec`` via
    exponential moving average (EMA).  After the E‑step is complete,
    the server broadcasts the updated weight vectors back to clients.

    Subclasses should implement ``_dispatch_weights`` to define how
    weight vectors are sent to clients for Full‑MoE or Fusion‑MoE.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mapping from client id -> weight vector [w_0,m, ..., w_{K-1,m}]
        self.client_weights: Dict[int, List[float]] = {}
        # EMA smoothing factor (0 <= beta < 1).  A higher beta means
        # slower adaptation to new accuracies.
        self.beta = float(getattr(self._cfg.llm.adapter, 'ema_beta', 0.9))
        # Prepare a buffer for per‑round per‑client accuracies
        self.msg_buffer['adapter_acc'] = {}

    # ------------------------------------------------------------------
    # Override grouping logic
    def _start_new_training_round(self, aggregated_num: int = 0,
                                  skip_grouping: bool = False):
        """
        Trigger a new round.  For MoE variants we replace the grouping
        trigger with a soft assignment step.  When the grouping period
        arrives, broadcast an ``adapter_eval`` message to all clients
        and wait for responses.  Otherwise, proceed with normal FL
        behaviour.
        """
        # Skip grouping logic when grouping is disabled or fixed
        if self._grouping_is_fixed:
            return super()._start_new_training_round(aggregated_num)

        # Only trigger soft grouping at specified intervals
        if self._cfg.llm.adapter.grouping.use and not skip_grouping:
            total_warmup_round = 0
            if self._cfg.llm.adapter.warmup.use:
                total_warmup_round = (self._cfg.llm.adapter.warmup.round
                                      * self._cfg.llm.adapter.count)
            regroup_interval = self._cfg.llm.adapter.grouping.round
            if (self.state >= total_warmup_round and
                    (self.state - total_warmup_round) % regroup_interval == 0):
                logger.info('Server: performing MoE E‑step (soft grouping)')
                # Broadcast evaluation request with the current model
                self.broadcast_model_para(msg_type='adapter_eval',
                                          filter_unseen_clients=False)
                return
        # Otherwise start a normal training round
        return super()._start_new_training_round(aggregated_num)

    def callback_funcs_for_grouping(self, message: Message) -> bool:
        """
        Callback invoked when a client sends its per‑adapter metrics.

        Each message is expected to have ``content`` of the form
        ``{'adapter_0_acc': float, 'adapter_1_acc': float, ...}``.
        We accumulate these per client and, once all clients have
        responded, compute new weight vectors and dispatch them.
        """
        if self._grouping_is_fixed:
            return False

        rnd = message.state
        cid = message.sender
        content: Dict[str, Any] = message.content
        # Initialise storage for this round
        if rnd not in self.msg_buffer['adapter_acc']:
            self.msg_buffer['adapter_acc'][rnd] = {}
        # Extract accuracies into a list sorted by adapter index
        K = int(self._cfg.llm.adapter.count)
        acc_list: List[float] = []
        for u in range(K):
            key = f'adapter_{u}_acc'
            acc_list.append(float(content.get(key, 0.0)))
        self.msg_buffer['adapter_acc'][rnd][cid] = acc_list

        # Check if all clients responded
        if len(self.msg_buffer['adapter_acc'][rnd]) < self.client_num:
            return False
        # Compute new weight vectors for each client
        client_accs = self.msg_buffer['adapter_acc'][rnd]
        for cid, accs in client_accs.items():
            total_acc = sum(accs)
            # If no accuracy reported (unlikely), use uniform
            if total_acc <= 0:
                accs = [1.0 for _ in accs]
                total_acc = sum(accs)
            accs_normalised = [a / total_acc for a in accs]
            if cid not in self.client_weights:
                # Initialise with uniform distribution
                w_old = [1.0 / len(accs) for _ in accs]
            else:
                w_old = self.client_weights[cid]
            # EMA update
            w_new = [self.beta * w_old[u] + (1.0 - self.beta) * accs_normalised[u]
                     for u in range(len(accs))]
            # Renormalise to ensure sum(w)=1
            s = sum(w_new)
            if s <= 0:
                s = 1.0
            w_new = [w / s for w in w_new]
            self.client_weights[cid] = w_new

        # Dispatch weight vectors to clients and proceed
        self._dispatch_weights(self.client_weights)
        # Clear buffer for this round
        del self.msg_buffer['adapter_acc'][rnd]
        # Start new training round (skip grouping)
        self._start_new_training_round(skip_grouping=True)
        return True

    def _dispatch_weights(self, w_map: Dict[int, List[float]]):
        """Dispatch updated weights to clients.

        Subclasses must override this method to send weight vectors
        appropriately.  It is responsible for constructing and sending
        ``set_adapter_weights`` messages to clients.
        """
        raise NotImplementedError


class FullMoEServer(_BaseMoEServer):
    """Server implementing soft assignment and dispatch for Full‑MoE."""

    def _dispatch_weights(self, w_map: Dict[int, List[float]]):
        for cid, w_vec in w_map.items():
            # send weight vector to each client; receiver must be a list
            self.comm_manager.send(Message(
                msg_type='set_adapter_weights',
                sender=self.ID,
                receiver=[cid],
                state=self.state,
                timestamp=self.cur_timestamp,
                content={'w_vec': w_vec}
            ))


class FusionMoEServer(_BaseMoEServer):
    """Server implementing soft assignment and dispatch for Fusion‑MoE."""

    def _dispatch_weights(self, w_map: Dict[int, List[float]]):
        for cid, w_vec in w_map.items():
            self.comm_manager.send(Message(
                msg_type='set_adapter_weights',
                sender=self.ID,
                receiver=[cid],
                state=self.state,
                timestamp=self.cur_timestamp,
                content={'w_vec': w_vec}
            ))


class _BaseMoEClient(LLMMultiLoRAClient):
    """Client base class supporting weight vector handling.

    Subclasses must override methods for evaluation and training to use
    ``self.w_vec`` as appropriate.  The base class adds a handler for
    ``set_adapter_weights`` messages and stores the received vector.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight vector initialised on server dispatch
        self.w_vec: Optional[List[float]] = None
        # Register handler for receiving weight vectors
        self.register_handlers('set_adapter_weights',
                               self.callback_funcs_for_set_adapter_weights,
                               [])

    def callback_funcs_for_set_adapter_weights(self, message: Message) -> bool:
        """Store the weight vector sent by the server."""
        w_vec = message.content.get('w_vec', None)
        if w_vec is None:
            logger.warning(f'Client {self.ID}: received empty w_vec')
        else:
            self.w_vec = list(map(float, w_vec))
            logger.info(f'Client {self.ID}: updated weight vector = {self.w_vec}')
        return True

    # ------------------------------------------------------------------
    # Override evaluation callback to compute accuracies (or rewards) per
    # adapter instead of only validation loss.  We reuse much of the
    # ``callback_funcs_for_adapter_eval`` logic from the base class.
    def callback_funcs_for_adapter_eval(self, message: Message) -> bool:
        """Evaluate each adapter on the validation set and send accuracies.

        This function is called when the server sends an ``adapter_eval``
        message.  We iterate over all adapters, compute classification
        accuracy on the validation set, and return the results as a
        dictionary ``{'adapter_{u}_acc': acc_u}``.  Accuracy is computed
        as the fraction of correctly classified samples based on the
        first valid token (similar to reward calculation in the
        ``RewardChoiceTrainer``).
        """
        if not hasattr(self, 'val_data') or self.val_data is None:
            # Fallback to training data if no explicit validation set
            self.val_data = self.data['val'] if 'val' in self.data else None
        if self.val_data is None:
            logger.warning(f'Client {self.ID}: no validation data; cannot compute accuracies')
            # send zeros to avoid blocking
            K = int(self.cfg.llm.adapter.count)
            res = {f'adapter_{u}_acc': 0.0 for u in range(K)}
            self.comm_manager.send(Message(
                msg_type='grouping',
                sender=self.ID,
                receiver=[0],
                state=self.state,
                timestamp=self.cur_timestamp,
                content=res
            ))
            return True

        # Evaluate each adapter
        from torch.utils.data import DataLoader
        import torch
        K = int(self.cfg.llm.adapter.count)
        adapter_names = [f'Adapter_{u}' for u in range(K)]
        # DataLoader for validation set
        loader = DataLoader(self.val_data, batch_size=self.cfg.dataloader.batch_size)
        acc_dict = {}
        for idx, adapter_name in enumerate(adapter_names):
            # Activate adapter
            self.model.set_active_adapter(adapter_name)
            correct = 0
            total = 0
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids,
                                         labels=labels,
                                         attention_mask=attention_mask)
                logits = outputs.logits
                # Compute classification accuracy as in RewardChoiceTrainer
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Candidate labels are provided as ``choices`` in config
                choices = self.cfg.trainer.choices
                choices_tensor = torch.tensor([ord(ch) - 65 for ch in choices], device=logits.device)
                new_labels = torch.full_like(shift_labels, -100)
                for j, c in enumerate(choices_tensor):
                    new_labels[shift_labels == c] = j
                restricted = shift_logits[..., choices_tensor]
                # Pick the first valid token for each sample
                flat_labels = new_labels.view(-1)
                flat_logits = restricted.view(-1, len(choices_tensor))
                keep = flat_labels != -100
                if keep.any():
                    preds = torch.argmax(flat_logits[keep], dim=-1)
                    labels_valid = flat_labels[keep]
                    correct += int((preds == labels_valid).sum().item())
                    total += len(labels_valid)
            acc = correct / total if total > 0 else 0.0
            acc_dict[f'adapter_{idx}_acc'] = float(acc)

        # Send results back to the server
        self.comm_manager.send(Message(
            msg_type='grouping',
            sender=self.ID,
            receiver=[0],  # always send to server 0
            state=self.state,
            timestamp=self.cur_timestamp,
            content=acc_dict
        ))
        return True


class FullMoEClient(_BaseMoEClient):
    """Client implementation for Full‑MoE.

    In Full‑MoE, each client trains all experts simultaneously.  The
    trainer must support the ``w_vec`` attribute on the context; this
    implementation attaches the current weight vector to the training
    context before each batch.
    """

    def callback_funcs_for_model_para(self, message: Message) -> bool:
        """Receive server model and run local training using w_vec."""
        # Call parent to set up model and sampler
        super().callback_funcs_for_model_para(message)
        # Ensure weight vector is available
        if self.w_vec is None:
            K = int(self.cfg.llm.adapter.count)
            self.w_vec = [1.0 / K for _ in range(K)]
        # Insert w_vec into context for the trainer to use
        if hasattr(self.trainer, 'ctx'):
            setattr(self.trainer.ctx, 'w_vec', self.w_vec)
        # Proceed with local training
        self.trainer.train()
        # After training, send all adapter params along with weights
        model_para = self.model.state_dict()
        # Include weight vector in feedback for aggregator
        self.comm_manager.send(Message(
            msg_type='model_para',
            sender=self.ID,
            receiver=[0],
            state=self.state,
            timestamp=self.cur_timestamp,
            content={'model_para': model_para, 'w': self.w_vec}
        ))
        return True


class FusionMoEClient(_BaseMoEClient):
    """Client implementation for Fusion‑MoE.

    In Fusion‑MoE, the client constructs a fused default adapter from
    all expert adapters using ``w_vec``, trains only the default
    adapter on local data, and sends back the updated default adapter
    along with the weight vector.
    """

    def _fuse_experts(self):
        """Fuse expert adapters into the default adapter using w_vec."""
        # Default adapter name (should exist)
        default_keys = [k for k in self.model.state_dict().keys() if 'default' in k]
        K = int(self.cfg.llm.adapter.count)
        # Construct fused parameter dict
        fused = {k: 0.0 for k in default_keys}
        # For each expert adapter, accumulate weighted params
        for u in range(K):
            w = self.w_vec[u] if self.w_vec and u < len(self.w_vec) else 1.0 / K
            prefix = f'Adapter_{u}'
            for k in default_keys:
                expert_key = k.replace('default', prefix)
                if expert_key in self.model.state_dict():
                    fused[k] = fused[k] + w * self.model.state_dict()[expert_key]
        # Assign fused parameters to default adapter
        state = self.model.state_dict()
        for k in default_keys:
            state[k] = fused[k]
        self.model.load_state_dict(state, strict=False)

    def callback_funcs_for_model_para(self, message: Message) -> bool:
        # Call parent to set up model
        super().callback_funcs_for_model_para(message)
        # Ensure w_vec is available
        if self.w_vec is None:
            K = int(self.cfg.llm.adapter.count)
            self.w_vec = [1.0 / K for _ in range(K)]
        # Fuse experts into default adapter
        self._fuse_experts()
        # Train only the default adapter (trainer handles this)
        if hasattr(self.trainer, 'ctx'):
            setattr(self.trainer.ctx, 'w_vec', self.w_vec)
        self.trainer.train()
        # After training, extract updated default adapter
        model_para = {}
        for k, v in self.model.state_dict().items():
            if 'default' in k:
                model_para[k] = v
        # Send updated default adapter and w_vec
        self.comm_manager.send(Message(
            msg_type='model_para',
            sender=self.ID,
            receiver=[0],
            state=self.state,
            timestamp=self.cur_timestamp,
            content={'model_para': model_para, 'w': self.w_vec}
        ))
        return True