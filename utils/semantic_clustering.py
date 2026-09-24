from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from utils.semantic_normalization import normalize_semantic_text

class SemanticTextEncoder:
    """Wrapper for text encoding, with TF-IDF fallback."""
    def __init__(self):
        self.backend = "tfidf"
        self.vectorizer = None
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        if self.backend == "exact_match" or self.vectorizer is None:
            # Fake embeddings if no sklearn
            return np.zeros((len(texts), 1))
            
        try:
            return self.vectorizer.fit_transform(texts).toarray()
        except Exception:
            # Fallback for empty vocabulary or errors
            return np.zeros((len(texts), 1))


@dataclass
class SemanticCluster:
    cluster_id: int
    item_indices: List[int]
    representative_label: str
    total_weight: float


def cluster_semantic_labels(
    labels: List[str],
    *,
    weights: List[float],
    encoder: SemanticTextEncoder,
    similarity_threshold: float = 0.80,
    representations: Optional[List[str]] = None,
    parents: Optional[List[str]] = None,
    concept_sets: Optional[List[set[str]]] = None,
    label_weight: float = 0.60,
    context_weight: float = 0.40,
    min_concept_overlap: float = 0.15,
) -> List[SemanticCluster]:
    """Cluster semantic labels using cosine similarity and weighted medoids."""
    if not labels:
        return []
        
    n = len(labels)
    normalized_labels = [normalize_semantic_text(l) for l in labels]
    
    # 1. Compute embeddings
    if encoder.backend == "exact_match":
        # Force exact match logic by setting sim = 1 if equal else 0
        sim_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if normalized_labels[i] != "unknown" and normalized_labels[i] == normalized_labels[j]:
                    sim_matrix[i, j] = 1.0
                    sim_matrix[j, i] = 1.0
    else:
        # TF-IDF or other continuous embedding
        embeddings = encoder.encode(normalized_labels)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized_embs = embeddings / norms
        label_sim_matrix = np.dot(normalized_embs, normalized_embs.T)
        
        if representations and parents and concept_sets:
            rep_embeddings = encoder.encode(representations)
            rep_norms = np.linalg.norm(rep_embeddings, axis=1, keepdims=True)
            rep_norms[rep_norms == 0] = 1e-9
            normalized_rep_embs = rep_embeddings / rep_norms
            rep_sim_matrix = np.dot(normalized_rep_embs, normalized_rep_embs.T)
            
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                        continue
                        
                    label_sim = label_sim_matrix[i, j]
                    context_sim = rep_sim_matrix[i, j]
                    combined_sim = label_weight * label_sim + context_weight * context_sim
                    
                    p_i = normalize_semantic_text(parents[i])
                    p_j = normalize_semantic_text(parents[j])
                    
                    ci = concept_sets[i]
                    cj = concept_sets[j]
                    overlap = len(ci & cj) / max(1, len(ci | cj))
                    
                    if label_sim >= similarity_threshold or (combined_sim >= similarity_threshold and p_i == p_j and overlap >= min_concept_overlap):
                        sim_matrix[i, j] = max(label_sim, combined_sim)
                    else:
                        sim_matrix[i, j] = 0.0
        else:
            sim_matrix = label_sim_matrix
        
    # 2. Agglomerative/Connected Components Clustering
    clusters = []
    visited = set()
    
    for i in range(n):
        if i in visited:
            continue
            
        if normalized_labels[i] == "unknown":
            clusters.append([i])
            visited.add(i)
            continue
            
        current_cluster = [i]
        visited.add(i)
        
        # Grow cluster
        to_check = [i]
        while to_check:
            curr = to_check.pop(0)
            for j in range(n):
                if j not in visited and normalized_labels[j] != "unknown":
                    if sim_matrix[curr, j] >= similarity_threshold:
                        visited.add(j)
                        current_cluster.append(j)
                        to_check.append(j)
                        
        clusters.append(current_cluster)
        
    # 3. Choose medoids and build SemanticCluster objects
    result = []
    for c_id, indices in enumerate(clusters):
        if normalized_labels[indices[0]] == "unknown":
            result.append(SemanticCluster(
                cluster_id=c_id,
                item_indices=indices,
                representative_label="unknown",
                total_weight=sum(weights[i] for i in indices)
            ))
            continue
            
        # Find weighted medoid
        best_label = ""
        max_score = -1.0
        
        # Fallback tie breaker
        for i in indices:
            label_i = normalized_labels[i]
            # Average weighted similarity to others in cluster
            score = 0.0
            for j in indices:
                score += sim_matrix[i, j] * weights[j]
                
            if score > max_score:
                max_score = score
                best_label = labels[i] # keep original formatting
            elif score == max_score:
                # Tie break 1: support (weight)
                if weights[i] > weights[indices[0]]: # Simplification for tie break
                    best_label = labels[i]
                elif weights[i] == weights[indices[0]] and labels[i] < best_label:
                    # Tie break 2: lexicographical
                    best_label = labels[i]
                    
        result.append(SemanticCluster(
            cluster_id=c_id,
            item_indices=indices,
            representative_label=best_label,
            total_weight=sum(weights[i] for i in indices)
        ))
        
    return result

@dataclass
class FieldAggregationResult:
    value: str
    agreement: float
    confidence: float
    alternative_values: List[Dict[str, Any]]
    winning_indices: List[int] = field(default_factory=list)


def weighted_semantic_field_vote(
    values: List[str],
    *,
    confidences: List[float],
    observation_weights: List[float],
    encoder: Optional[SemanticTextEncoder] = None,
    similarity_threshold: float = 0.80,
    exact_match: bool = False,
    alternative_support_threshold: float = 0.20,
    representations: Optional[List[str]] = None,
    parents: Optional[List[str]] = None,
    concept_sets: Optional[List[set[str]]] = None,
    label_weight: float = 0.60,
    context_weight: float = 0.40,
    min_concept_overlap: float = 0.15,
) -> FieldAggregationResult:
    """Aggregates a single target field."""
    if not values:
        return FieldAggregationResult("unknown", 0.0, 0.0, [])
        
    n = len(values)
    
    # Calculate effective weights (conf * obs)
    effective_weights = [confidences[i] * observation_weights[i] for i in range(n)]
    total_weight = sum(effective_weights)
    
    if total_weight == 0:
        return FieldAggregationResult("unknown", 0.0, 0.0, [])
        
    if exact_match or encoder is None:
        # Simple exact match weighted voting
        votes = {}
        conf_sums = {}
        obs_sums = {}
        
        for i, val in enumerate(values):
            norm_val = normalize_semantic_text(val)
            votes[norm_val] = votes.get(norm_val, 0.0) + effective_weights[i]
            conf_sums[norm_val] = conf_sums.get(norm_val, 0.0) + effective_weights[i]
            obs_sums[norm_val] = obs_sums.get(norm_val, 0.0) + observation_weights[i]
            
        winning_val, winning_weight = max(votes.items(), key=lambda x: x[1])
        
        winning_indices = [i for i, val in enumerate(values) if normalize_semantic_text(val) == winning_val]
        
        agg_conf = (conf_sums[winning_val] / obs_sums[winning_val]) if obs_sums[winning_val] > 0 else 0.0
        agreement = winning_weight / total_weight
        
        alts = []
        for k, v in votes.items():
            support = v / total_weight
            if k != winning_val and support >= alternative_support_threshold and k != "unknown":
                alt_conf = (conf_sums[k] / obs_sums[k]) if obs_sums[k] > 0 else 0.0
                alts.append({
                    "label": k, 
                    "support": float(support),
                    "confidence": float(alt_conf)
                })
                
        alts.sort(key=lambda x: (-x["support"], x["label"]))
                
        return FieldAggregationResult(winning_val, float(agreement), float(agg_conf), alts, winning_indices)
        
    # Clustering based vote
    clusters = cluster_semantic_labels(
        values,
        weights=effective_weights,
        encoder=encoder,
        similarity_threshold=similarity_threshold,
        representations=representations,
        parents=parents,
        concept_sets=concept_sets,
        label_weight=label_weight,
        context_weight=context_weight,
        min_concept_overlap=min_concept_overlap
    )
    
    winning_cluster = max(clusters, key=lambda c: c.total_weight)
    
    # Calculate agg_conf for winning cluster
    cluster_conf_sum = sum(confidences[i] * observation_weights[i] for i in winning_cluster.item_indices)
    cluster_obs_sum = sum(observation_weights[i] for i in winning_cluster.item_indices)
    
    agg_conf = (cluster_conf_sum / cluster_obs_sum) if cluster_obs_sum > 0 else 0.0
    agreement = winning_cluster.total_weight / total_weight
    
    alts = []
    for c in clusters:
        if c.cluster_id != winning_cluster.cluster_id:
            support = c.total_weight / total_weight
            if support >= alternative_support_threshold and c.representative_label != "unknown":
                alt_conf_sum = sum(confidences[i] * observation_weights[i] for i in c.item_indices)
                alt_obs_sum = sum(observation_weights[i] for i in c.item_indices)
                alt_conf = (alt_conf_sum / alt_obs_sum) if alt_obs_sum > 0 else 0.0
                alts.append({
                    "label": c.representative_label, 
                    "support": float(support),
                    "confidence": float(alt_conf)
                })
                
    alts.sort(key=lambda x: (-x["support"], x["label"]))
                
    return FieldAggregationResult(
        winning_cluster.representative_label,
        float(agreement),
        float(agg_conf),
        alts,
        winning_cluster.item_indices
    )

MEASURE_LABEL_NORMALIZATION = {
    "accumulated amount": "accumulated",
    "total amount": "total",
    "average value": "average",
}

def build_target_label(
    primary_entity: str,
    measure: str,
    temporal_granularity: str,
) -> str:
    """Build a readable target label deterministically."""
    parts = []
    
    def add_if_known(val: str, normalize: bool = False):
        if val and val.lower() != "unknown":
            norm = val.lower().strip()
            if normalize and norm in MEASURE_LABEL_NORMALIZATION:
                norm = MEASURE_LABEL_NORMALIZATION[norm]
            if norm not in parts:
                parts.append(norm)
            
    add_if_known(temporal_granularity)
    add_if_known(measure, normalize=True)
    add_if_known(primary_entity)
    
    if not parts:
        return "unknown"
        
    return " ".join(parts).capitalize()


def build_likely_target(
    *,
    primary_entity: str,
    measure: str,
    temporal_granularity: str,
) -> str:
    """Build a readable likely target deterministically."""
    return build_target_label(primary_entity, measure, temporal_granularity)
