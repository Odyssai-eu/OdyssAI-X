"""Corrige PipelineMixin.pipeline : des couches n'etaient calculees par PERSONNE.

Bug amont (mlx-lm 0.31.3 ET le fork ds4 — les deux fichiers sont identiques).
`pipeline()` repartit les couches a l'envers (rank 0 = les DERNIERES) et donne
une couche de plus aux `extra` premiers rangs quand le total n'est pas
divisible. Mais il calcule la borne avec le `layers_per_rank` DEJA incremente
du rang courant :

    layers_per_rank = len(self.layers) // self.pipeline_size
    extra = len(self.layers) - layers_per_rank * self.pipeline_size
    if self.pipeline_rank < extra:
        layers_per_rank += 1
    self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank

Les rangs n'ont alors pas tous le meme pas, les tranches ne se raccordent plus,
et un trou apparait entre la tranche du rang 0 et celle du rang 1. Le modele
tourne quand meme (aucune exception : les couches orphelines sont juste
absentes des deux cotes) et produit du texte fluide — la panne est SILENCIEUSE.

Verifie a l'execution sur les vraies classes, 2026-08-03 :
  * DeepSeek-V4-Flash, 43 couches / 2 rangs -> couche 21 jamais calculee (42/43)
  * DeepSeek-V4-Pro,   61 couches / 5 rangs -> couches 48..51 jamais calculees (57/61)
  * un total divisible (60/5) est correct — d'ou le fait que ca soit passe
    inapercu sur les modeles a compte de couches rond.

Le correctif enchaine les bornes de facon cumulative depuis la fin, ce qui
garantit par construction : union des tranches = toutes les couches, et
intersection vide. Conserve la convention amont (rang 0 = dernieres couches),
donc rien d'autre ne bouge : les taps DSpark restent locaux a rank0 et le
sens du send/recv est inchange.
"""

from mlx_lm.models.pipeline import PipelineMixin


def _pipeline_fixed(self, group):
    self.pipeline_rank = group.rank()
    self.pipeline_size = group.size()
    n = len(self.layers)
    base, extra = divmod(n, self.pipeline_size)
    # Taille de chaque rang : les `extra` premiers en portent une de plus.
    sizes = [base + (1 if r < extra else 0) for r in range(self.pipeline_size)]
    # Rang 0 prend le bloc de fin, rang 1 le precedent, etc. En chainant les
    # bornes on ne peut ni laisser de trou ni se chevaucher.
    end = n - sum(sizes[: self.pipeline_rank])
    start = end - sizes[self.pipeline_rank]
    self.start_idx = start
    self.end_idx = end
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx


def apply_pipeline_split_fix() -> None:
    if getattr(PipelineMixin.pipeline, "_odyssai_split_fix", False):
        return
    _pipeline_fixed._odyssai_split_fix = True
    PipelineMixin.pipeline = _pipeline_fixed
