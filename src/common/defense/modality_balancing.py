# coding: utf-8
"""Modality Balancing defense module (MM'23)."""

from contextlib import contextmanager
from logging import getLogger
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


_MODALITY_ALIASES = {
    'visual': 'visual',
    'vision': 'visual',
    'image': 'visual',
    'textual': 'textual',
    'text': 'textual',
}


@contextmanager
def override_modality_features(model, modality: str, new_features: torch.Tensor):
    """Temporarily override the model's modality features."""
    original = model.get_modal_features(modality)
    model.set_modal_features(modality, new_features)
    try:
        yield
    finally:
        model.set_modal_features(modality, original)


class ModalityBalancing:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.defense_cfg = config['defense'] or {}
        self.attack_cfg = config['attack'] or {}

    def enabled(self) -> bool:
        return bool(self.defense_cfg.get('enable', False))

    def _normalize_modality(self, modality: str) -> str:
        if modality is None:
            return None
        return _MODALITY_ALIASES.get(modality, modality)

    def distill_adv_embedding(
        self,
        modality: str,
        pos_item_id: torch.Tensor,
        neg_item_id: torch.Tensor,
    ) -> torch.Tensor:
        """PGD distillation in feature space for a single (i, j)."""
        modality = self._normalize_modality(modality)
        eps = self.defense_cfg.get('distill_eps', 1.0)
        pgd_steps = self.defense_cfg.get('pgd_steps', 10)
        step_size = self.defense_cfg.get('pgd_step_size')
        if step_size is None:
            step_size = eps / max(pgd_steps, 1)

        raw_features = self.model.get_modal_features(modality)
        x_orig = raw_features[pos_item_id].detach()
        z = x_orig.clone().detach()

        with torch.no_grad():
            neg_embed = self.model.encode_item_feature(modality, neg_item_id)

        for _ in range(pgd_steps):
            z = z.detach().clone().requires_grad_(True)
            pos_embed = self.model.encode_item_feature(
                modality,
                pos_item_id,
                feature_override={'item_id': pos_item_id, 'feature': z},
            )
            distill_loss = ((pos_embed - neg_embed) ** 2).sum()
            grad = torch.autograd.grad(distill_loss, z, retain_graph=False, create_graph=False)[0]
            z = z - step_size * grad.sign()
            z = torch.max(torch.min(z, x_orig + eps), x_orig - eps)

        z = z.detach()
        adv_embed = self.model.encode_item_feature(
            modality,
            pos_item_id,
            feature_override={'item_id': pos_item_id, 'feature': z},
        )
        return adv_embed

    def compute_balance_loss(
        self,
        sensitive_modality: str,
        robust_modality: str,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Modality Balancing loss for sampled triples."""
        sensitive_modality = self._normalize_modality(sensitive_modality)
        robust_modality = self._normalize_modality(robust_modality)

        e_u_s, e_i_s = self.model.get_user_item_embeddings(sensitive_modality, user_ids, pos_item_ids)
        _, e_j_s = self.model.get_user_item_embeddings(sensitive_modality, user_ids, neg_item_ids)
        e_u_r, e_i_r = self.model.get_user_item_embeddings(robust_modality, user_ids, pos_item_ids)
        _, e_j_r = self.model.get_user_item_embeddings(robust_modality, user_ids, neg_item_ids)

        adv_embeds = []
        for pos_id, neg_id in zip(pos_item_ids, neg_item_ids):
            adv_embeds.append(self.distill_adv_embedding(sensitive_modality, pos_id, neg_id))
        e_i_adv = torch.stack(adv_embeds, dim=0)

        s_margin_r = (e_u_r * e_i_r).sum(dim=1) - (e_u_r * e_j_r).sum(dim=1)
        s_margin_s = (e_u_s * e_i_s).sum(dim=1) - (e_u_s * e_i_adv).sum(dim=1)

        balance_loss = F.relu(s_margin_s - s_margin_r)
        return balance_loss.mean()

    def apply_modality_balancing_step(
        self,
        interaction: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Sample N triplets from current batch and compute balance loss."""
        if not self.enabled():
            return None

        sensitive_modality = self.defense_cfg.get('sensitive_modality')
        robust_modality = self.defense_cfg.get('robust_modality')
        if sensitive_modality is None or robust_modality is None:
            return None

        user_ids = interaction[0]
        pos_item_ids = interaction[1]
        neg_item_ids = interaction[2]
        batch_size = user_ids.shape[0]
        sample_num = int(self.defense_cfg.get('N', 20))
        if batch_size == 0 or sample_num <= 0:
            return None
        indices = torch.randint(0, batch_size, (sample_num,), device=user_ids.device)

        return self.compute_balance_loss(
            sensitive_modality,
            robust_modality,
            user_ids[indices],
            pos_item_ids[indices],
            neg_item_ids[indices],
        )

    def generate_attack_delta(
        self,
        eval_data,
        modality: str,
    ) -> Optional[torch.Tensor]:
        """Generate FGSM-L2 perturbation for a modality."""
        modality = self._normalize_modality(modality)
        features = self.model.get_modal_features(modality)
        if features is None:
            return None

        eps_ratio = float(self.attack_cfg.get('eps_ratio', 0.05))
        sample_size = int(self.attack_cfg.get('sample_size', 1024))

        user_ids, pos_item_ids, neg_item_ids = self._sample_attack_triples(eval_data, sample_size)
        if user_ids is None:
            return None

        delta = torch.zeros_like(features, requires_grad=True)
        # trainer.evaluate runs under torch.no_grad(), so enable grad here to compute
        # gradients w.r.t. the input perturbation delta.
        with torch.enable_grad():
            with override_modality_features(self.model, modality, features + delta):
                user_embed, item_embed = self.model.get_fused_embeddings()
                pos_scores = (user_embed[user_ids] * item_embed[pos_item_ids]).sum(dim=1)
                neg_scores = (user_embed[user_ids] * item_embed[neg_item_ids]).sum(dim=1)
                loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        grad_norm = grad.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        eps = eps_ratio * features.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        delta = eps * grad / grad_norm
        return delta.detach()

    def _sample_attack_triples(self, eval_data, sample_size: int):
        dataset = eval_data.dataset
        if len(dataset) == 0:
            return None, None, None
        total = len(dataset)
        sample_size = min(sample_size, total)
        idx = torch.randint(0, total, (sample_size,), device=self.config['device'])
        sampled = dataset.df.iloc[idx.cpu().numpy()]
        user_ids = torch.tensor(sampled[dataset.uid_field].values, device=self.config['device'])
        pos_item_ids = torch.tensor(sampled[dataset.iid_field].values, device=self.config['device'])

        all_items = torch.arange(dataset.get_item_num(), device=self.config['device'])
        history = self._build_history(eval_data)
        neg_items = []
        for u in user_ids.tolist():
            neg = all_items[torch.randint(0, all_items.shape[0], (1,))].item()
            if u in history:
                while neg in history[u]:
                    neg = all_items[torch.randint(0, all_items.shape[0], (1,))].item()
            neg_items.append(neg)
        neg_item_ids = torch.tensor(neg_items, device=self.config['device'])
        return user_ids, pos_item_ids, neg_item_ids

    def _build_history(self, eval_data) -> Dict[int, set]:
        if hasattr(eval_data, '_attack_history_cache'):
            return eval_data._attack_history_cache
        history = {}
        uid_field = eval_data.additional_dataset.uid_field
        iid_field = eval_data.additional_dataset.iid_field
        uid_freq = eval_data.additional_dataset.df.groupby(uid_field)[iid_field]
        for u, items in uid_freq:
            history[u] = set(items.values)
        eval_data._attack_history_cache = history
        return history

    def evaluate_attack(self, trainer, eval_data) -> Dict[str, Dict[str, float]]:
        """Evaluate attack metrics for each modality."""
        results = {}
        for modality in self.model.get_modalities():
            delta = self.generate_attack_delta(eval_data, modality)
            if delta is None:
                continue
            features = self.model.get_modal_features(modality)
            with override_modality_features(self.model, modality, features + delta):
                self.model.prepare_full_sort()
                attack_result = trainer.evaluate(eval_data, is_test=True, apply_attack=False)
            results[modality] = attack_result
        return results
