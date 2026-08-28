
"""
Construire un fichier features.csv combinant :
- des variables dérivées des personnages à partir de fichiers .book
- des variables de métadonnées (canon, genre littéraire, date, genre de l'auteur, etc.)
- des variables lexicales spécifiques (logit_share_female_<role>_<word>)

À la fin, on ne garde que les colonnes dont moins de 99 % des valeurs sont manquantes.
"""

import ast
import math
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Paramètres de base
# -------------------------------------------------------------------

BOOK_DIR = Path("OUTPUT_BOOK_GEN_Z")                
META_PATH = Path("CHAPITRES_METADATA.csv")
OUTPUT_PATH = Path("features.csv")

# Petit epsilon pour éviter les divisions par zéro
EPS = 1e-9

# Lexiques simples pour quelques features thématiques
APPEARANCE_ADJS = {
    "beau", "belle", "joli", "jolie", "jeune", "vieux", "vieille", "pâle", "pale"
}
STRENGTH_ADJS = {
    "fort", "forte", "puissant", "puissante", "courageux", "courageuse", "brave"
}
DOMESTIC_NOUNS = {
    "maison", "foyer", "famille", "enfant", "enfants", "mère", "mere", "père", "pere",
    "femme", "épouse", "epouse"
}
WEAPON_NOUNS = {
    "épée", "epee", "pistolet", "arme", "armes", "sabre", "fusil", "canon"
}

# Seuil minimal d'occurrences (Female + Male) par rôle et par mot
# pour créer logit_share_female_<role>_<word>
MIN_ROLE_WORD_TOTAL = 30


# -------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------

def gini(values: List[float]) -> float:
    """Coefficient de Gini sur des valeurs positives (ex : nb de mentions)."""
    arr = np.array(values, dtype=float)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return np.nan
    if np.all(arr == 0):
        return 0.0
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    g = (2.0 * np.sum((np.arange(1, n + 1) * arr_sorted)) /
         (n * np.sum(arr_sorted))) - (n + 1) / n
    return float(g)


def shannon_entropy(counts: List[float]) -> float:
    """Entropie de Shannon sur des comptes (base e)."""
    arr = np.array(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return np.nan
    p = arr / arr.sum()
    return float(-np.sum(p * np.log(p)))


def safe_gender_label(char: Dict[str, Any]) -> str:
    """Renvoie 'Male', 'Female' ou 'Unknown' selon char['gender']['argmax']."""
    g = char.get("gender", {}).get("argmax")
    if g in ("Male", "Female"):
        return g
    return "Unknown"


# -------------------------------------------------------------------
# Lecture d’un .book
# -------------------------------------------------------------------

def load_book(path: Path) -> List[Dict[str, Any]]:
    chars: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chars.append(ast.literal_eval(line))
            except Exception:
                continue
    return chars


# -------------------------------------------------------------------
# Extraction de features pour un livre
# -------------------------------------------------------------------

def extract_features_for_book(book_id: str, book_path: Path) -> Dict[str, Any]:
    """
    Extraire un ensemble de features au niveau du livre à partir du fichier .book.
    Chaque ligne du .book est un personnage (dict Python).
    """
    chars = load_book(book_path)
    if not chars:
        return {"book_id": book_id}

    gender_labels = [safe_gender_label(c) for c in chars]

    # -------------------
    # Comptages de base
    # -------------------
    n_characters = len(chars)

    n_male_chars = sum(g == "Male" for g in gender_labels)
    n_female_chars = sum(g == "Female" for g in gender_labels)
    n_unknown_chars = sum(g == "Unknown" for g in gender_labels)

    occ = np.array([c.get("count", {}).get("occurrence", 0) for c in chars], dtype=float)
    total_mentions = float(occ.sum()) if occ.size > 0 else 0.0

    occ_male = float(occ[[g == "Male" for g in gender_labels]].sum()) if total_mentions > 0 else 0.0
    occ_female = float(occ[[g == "Female" for g in gender_labels]].sum()) if total_mentions > 0 else 0.0
    occ_unknown = float(occ[[g == "Unknown" for g in gender_labels]].sum()) if total_mentions > 0 else 0.0

    order = np.argsort(-occ)  # indices triés par mentions décroissantes
    top1_idx = int(order[0]) if occ.size > 0 else None
    top3_idx = order[:3] if occ.size >= 3 else order
    top5_idx = order[:5] if occ.size >= 5 else order

    feats: Dict[str, Any] = {
        "book_id": book_id,
        "n_characters": n_characters,
        "share_male_chars": n_male_chars / n_characters if n_characters > 0 else np.nan,
        "share_female_chars": n_female_chars / n_characters if n_characters > 0 else np.nan,
        "share_nullgender_chars": n_unknown_chars / n_characters if n_characters > 0 else np.nan,
        "female_mentions_share": occ_female / total_mentions if total_mentions > 0 else np.nan,
        "male_mentions_share": occ_male / total_mentions if total_mentions > 0 else np.nan,
        "null_mentions_share": occ_unknown / total_mentions if total_mentions > 0 else np.nan,
        "total_mentions": total_mentions,
    }

    # Part des femmes dans top1 / top3 / top5 (mention-pondéré)
    if top1_idx is not None:
        feats["top1_char_is_female"] = 1.0 if gender_labels[top1_idx] == "Female" else 0.0
    else:
        feats["top1_char_is_female"] = np.nan

    if top3_idx.size > 0:
        top3_genders = [gender_labels[i] for i in top3_idx]
        feats["top3_share_female_chars"] = sum(g == "Female" for g in top3_genders) / len(top3_genders)
    else:
        feats["top3_share_female_chars"] = np.nan

    if top5_idx.size > 0:
        top5_mentions = occ[top5_idx].sum()
        top5_female_mentions = occ[
            [i for i in top5_idx if gender_labels[i] == "Female"]
        ].sum()
        feats["top5_female_mentions_share"] = (
            top5_female_mentions / top5_mentions if top5_mentions > 0 else np.nan
        )
    else:
        feats["top5_female_mentions_share"] = np.nan

    if occ.size > 0:
        feats["mean_mentions_per_char"] = float(occ.mean())
        feats["sd_mentions_per_char"] = float(occ.std())
    else:
        feats["mean_mentions_per_char"] = np.nan
        feats["sd_mentions_per_char"] = np.nan

    # -------------------
    # Qualité inférence de genre
    # -------------------
    gender_max = []
    gender_ratio_vals = []
    for c, g in zip(chars, gender_labels):
        if g in ("Male", "Female"):
            val = c.get("gender", {}).get("max")
            if isinstance(val, (int, float)):
                gender_max.append(float(val))
            ratio = c.get("gender", {}).get("ratio")
            if isinstance(ratio, (int, float)):
                gender_ratio_vals.append(float(ratio))

    if gender_max:
        feats["mean_gender_max"] = float(np.mean(gender_max))
        feats["sd_gender_max"] = float(np.std(gender_max))
        feats["share_lowconf_gender"] = float(np.mean(np.array(gender_max) < 0.7))
    else:
        feats["mean_gender_max"] = np.nan
        feats["sd_gender_max"] = np.nan
        feats["share_lowconf_gender"] = np.nan

    feats["mean_gender_ratio"] = float(np.mean(gender_ratio_vals)) if gender_ratio_vals else np.nan

    # -------------------
    # Rôles agent / patient / mod / poss
    # -------------------
    def count_role_tokens(role: str, target_gender: str) -> int:
        total = 0
        for c, g in zip(chars, gender_labels):
            if g != target_gender:
                continue
            tokens = c.get(role, [])
            total += len(tokens)
        return total

    female_agent_tokens = count_role_tokens("agent", "Female")
    male_agent_tokens = count_role_tokens("agent", "Male")
    female_patient_tokens = count_role_tokens("patient", "Female")
    male_patient_tokens = count_role_tokens("patient", "Male")
    female_mod_tokens = count_role_tokens("mod", "Female")
    male_mod_tokens = count_role_tokens("mod", "Male")
    female_poss_tokens = count_role_tokens("poss", "Female")
    male_poss_tokens = count_role_tokens("poss", "Male")

    total_agent_tokens = female_agent_tokens + male_agent_tokens
    total_patient_tokens = female_patient_tokens + male_patient_tokens
    total_mod_tokens = female_mod_tokens + male_mod_tokens
    total_poss_tokens = female_poss_tokens + male_poss_tokens

    feats.update({
        "female_agent_tokens": float(female_agent_tokens),
        "male_agent_tokens": float(male_agent_tokens),
        "female_patient_tokens": float(female_patient_tokens),
        "male_patient_tokens": float(male_patient_tokens),
        "female_mod_tokens": float(female_mod_tokens),
        "male_mod_tokens": float(male_mod_tokens),
        "female_poss_tokens": float(female_poss_tokens),
        "male_poss_tokens": float(male_poss_tokens),

        "share_female_agent_tokens": female_agent_tokens / total_agent_tokens if total_agent_tokens > 0 else np.nan,
        "share_female_patient_tokens": female_patient_tokens / total_patient_tokens if total_patient_tokens > 0 else np.nan,
        "share_female_mod_tokens": female_mod_tokens / total_mod_tokens if total_mod_tokens > 0 else np.nan,
        "share_female_poss_tokens": female_poss_tokens / total_poss_tokens if total_poss_tokens > 0 else np.nan,
    })

    ratio_agent_patient_female = (
        female_agent_tokens / (female_patient_tokens + EPS)
        if (female_agent_tokens + female_patient_tokens) > 0 else np.nan
    )
    ratio_agent_patient_male = (
        male_agent_tokens / (male_patient_tokens + EPS)
        if (male_agent_tokens + male_patient_tokens) > 0 else np.nan
    )

    feats["agent_patient_ratio_female"] = ratio_agent_patient_female
    feats["agent_patient_ratio_male"] = ratio_agent_patient_male
    if not math.isnan(ratio_agent_patient_female) and not math.isnan(ratio_agent_patient_male):
        feats["diff_agent_patient_ratio"] = ratio_agent_patient_female - ratio_agent_patient_male
    else:
        feats["diff_agent_patient_ratio"] = np.nan

    # -------------------
    # Descriptions (modificateurs) : mod par mention
    # -------------------
    mentions_female = occ[[g == "Female" for g in gender_labels]].sum()
    mentions_male = occ[[g == "Male" for g in gender_labels]].sum()

    mean_mod_per_mention_female = (
        female_mod_tokens / (mentions_female + EPS) if mentions_female > 0 else np.nan
    )
    mean_mod_per_mention_male = (
        male_mod_tokens / (mentions_male + EPS) if mentions_male > 0 else np.nan
    )

    feats["mean_mod_per_mention_female"] = mean_mod_per_mention_female
    feats["mean_mod_per_mention_male"] = mean_mod_per_mention_male
    if not math.isnan(mean_mod_per_mention_female) and not math.isnan(mean_mod_per_mention_male):
        feats["diff_mod_per_mention"] = mean_mod_per_mention_female - mean_mod_per_mention_male
    else:
        feats["diff_mod_per_mention"] = np.nan

    # -------------------
    # Champs lexicaux simples
    # -------------------
    def count_lex_from_role(role: str, target_gender: str, lexicon: set) -> int:
        total = 0
        for c, g in zip(chars, gender_labels):
            if g != target_gender:
                continue
            tokens = c.get(role, [])
            for t in tokens:
                w = str(t.get("w", "")).lower()
                if w in lexicon:
                    total += 1
        return total

    female_mod_appearance = count_lex_from_role("mod", "Female", APPEARANCE_ADJS)
    male_mod_strength = count_lex_from_role("mod", "Male", STRENGTH_ADJS)
    female_poss_domestic = count_lex_from_role("poss", "Female", DOMESTIC_NOUNS)
    male_poss_weapon = count_lex_from_role("poss", "Male", WEAPON_NOUNS)

    feats["female_mod_appearance_share"] = (
        female_mod_appearance / female_mod_tokens if female_mod_tokens > 0 else np.nan
    )
    feats["male_mod_strength_share"] = (
        male_mod_strength / male_mod_tokens if male_mod_tokens > 0 else np.nan
    )
    feats["female_poss_domestic_share"] = (
        female_poss_domestic / female_poss_tokens if female_poss_tokens > 0 else np.nan
    )
    feats["male_poss_weapon_share"] = (
        male_poss_weapon / male_poss_tokens if male_poss_tokens > 0 else np.nan
    )

    # -------------------
    # Pronoms et formes d'adresse
    # -------------------
    pron_counts: Dict[str, int] = {}
    n_chars_with_monsieur = 0
    n_chars_with_madame = 0

    for c in chars:
        mentions = c.get("mentions", {})
        for pr in mentions.get("pronoun", []):
            form = str(pr.get("n", "")).lower()
            cnt = int(pr.get("c", 0))
            pron_counts[form] = pron_counts.get(form, 0) + cnt

        common_mentions = mentions.get("common", [])
        has_monsieur = any("monsieur" in str(m.get("n", "")).lower() for m in common_mentions)
        has_madame = any("madame" in str(m.get("n", "")).lower() for m in common_mentions)
        if has_monsieur:
            n_chars_with_monsieur += 1
        if has_madame:
            n_chars_with_madame += 1

    feats["n_chars_with_monsieur"] = float(n_chars_with_monsieur)
    feats["n_chars_with_madame"] = float(n_chars_with_madame)

    il = pron_counts.get("il", 0) + pron_counts.get("ils", 0)
    elle = pron_counts.get("elle", 0) + pron_counts.get("elles", 0)
    je = pron_counts.get("je", 0) + pron_counts.get("j'", 0)
    total_pron_3rd = il + elle

    feats["pron_il"] = float(il)
    feats["pron_elle"] = float(elle)
    feats["pron_je"] = float(je)
    feats["elle_over_il"] = elle / (il + EPS) if il > 0 else np.nan
    feats["je_over_3rdperson"] = je / (total_pron_3rd + EPS) if total_pron_3rd > 0 else np.nan

    # -------------------
    # Position des personnages et inégalités de mentions
    # -------------------
    first_idx = []
    for c in chars:
        indices = []
        for role in ("agent", "patient", "mod", "poss"):
            for t in c.get(role, []):
                if "i" in t:
                    indices.append(int(t["i"]))
        if indices:
            first_idx.append(min(indices))
        else:
            first_idx.append(None)

    all_indices = []
    for c in chars:
        for role in ("agent", "patient", "mod", "poss"):
            for t in c.get(role, []):
                if "i" in t:
                    all_indices.append(int(t["i"]))
    max_idx = max(all_indices) if all_indices else None

    if max_idx is not None and any(i is not None for i in first_idx):
        valid = [(idx, g) for idx, g in zip(first_idx, gender_labels) if idx is not None]
        if valid:
            valid.sort(key=lambda x: x[0])
            feats["earliest_char_gender"] = valid[0][1]
        else:
            feats["earliest_char_gender"] = np.nan

        earliest_sorted = sorted(valid, key=lambda x: x[0])[:3]
        if earliest_sorted:
            feats["earliest3_share_female"] = sum(g == "Female" for _, g in earliest_sorted) / len(earliest_sorted)
        else:
            feats["earliest3_share_female"] = np.nan

        female_pos = [idx / max_idx for idx, g in zip(first_idx, gender_labels)
                      if idx is not None and g == "Female"]
        male_pos = [idx / max_idx for idx, g in zip(first_idx, gender_labels)
                    if idx is not None and g == "Male"]

        if female_pos:
            feats["median_intro_female"] = float(np.median(female_pos))
        else:
            feats["median_intro_female"] = np.nan
        if male_pos:
            feats["median_intro_male"] = float(np.median(male_pos))
        else:
            feats["median_intro_male"] = np.nan

        if female_pos and male_pos:
            feats["diff_introduction_position"] = feats["median_intro_female"] - feats["median_intro_male"]
        else:
            feats["diff_introduction_position"] = np.nan
    else:
        feats["earliest_char_gender"] = np.nan
        feats["earliest3_share_female"] = np.nan
        feats["median_intro_female"] = np.nan
        feats["median_intro_male"] = np.nan
        feats["diff_introduction_position"] = np.nan

    feats["gini_mentions"] = gini(list(occ))
    feats["entropy_character_mentions"] = shannon_entropy(list(occ))

    female_occ = occ[[g == "Female" for g in gender_labels]]
    male_occ = occ[[g == "Male" for g in gender_labels]]
    feats["gini_mentions_female"] = gini(list(female_occ)) if female_occ.size > 0 else np.nan
    feats["gini_mentions_male"] = gini(list(male_occ)) if male_occ.size > 0 else np.nan

    feats["entropy_gender_mentions"] = shannon_entropy(
        [occ_male, occ_female, occ_unknown]
    )

    # -------------------
    # Variables lexicales par mot et par rôle (génériques, logit + share)
    # -------------------
    agent_female = Counter()
    agent_male = Counter()
    mod_female = Counter()
    mod_male = Counter()
    poss_female = Counter()
    poss_male = Counter()

    for c, g in zip(chars, gender_labels):
        if g not in ("Male", "Female"):
            continue

        # agent
        for t in c.get("agent", []) or []:
            w = str(t.get("w", "")).lower()
            if not w:
                continue
            if g == "Female":
                agent_female[w] += 1
            else:
                agent_male[w] += 1

        # mod
        for t in c.get("mod", []) or []:
            w = str(t.get("w", "")).lower()
            if not w:
                continue
            if g == "Female":
                mod_female[w] += 1
            else:
                mod_male[w] += 1

        # poss
        for t in c.get("poss", []) or []:
            w = str(t.get("w", "")).lower()
            if not w:
                continue
            if g == "Female":
                poss_female[w] += 1
            else:
                poss_male[w] += 1

    def add_role_word_features(role: str,
                               female_counter: Counter,
                               male_counter: Counter,
                               feats_dict: Dict[str, Any]) -> None:
        all_words = set(female_counter) | set(male_counter)
        for w in all_words:
            fem = female_counter[w]
            male = male_counter[w]
            total = fem + male
            if total < MIN_ROLE_WORD_TOTAL:
                continue

            share = fem / total
            p = (fem + 0.5) / (total + 1.0)
            logit = float(np.log(p / (1.0 - p)))

            feats_dict[f"share_female_{role}_{w}"] = share
            feats_dict[f"logit_share_female_{role}_{w}"] = logit

    add_role_word_features("agent", agent_female, agent_male, feats)
    add_role_word_features("mod", mod_female, mod_male, feats)
    add_role_word_features("poss", poss_female, poss_male, feats)

    return feats


# -------------------------------------------------------------------
# Intégration des métadonnées (avec genre de l'auteur)
# -------------------------------------------------------------------

def add_metadata_features(features_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajouter des features de métadonnées :
    - gender (genre de l'auteur, tel que dans CHAPITRES_METADATA.csv)
    - meta_year (date numérique)
    - meta_canon (0/1)
    - dummies de genre littéraire : meta_genre_*
    On matche book_id (index de features_df) avec doc_name (sans suffixe .book).
    """
    meta_df = meta_df.copy()

    if "doc_name" not in meta_df.columns:
        print("[WARN] 'doc_name' absent des métadonnées : pas d'enrichissement.")
        return features_df

    # Créer un book_id cohérent avec les .book (on enlève l'extension .book)
    meta_df["book_id"] = meta_df["doc_name"].str.replace(r"\.book$", "", regex=True)
    meta_df = meta_df.set_index("book_id")

    # Sous-ensemble des colonnes utiles si elles existent
    cols_to_keep = [c for c in ["date", "canon", "genre", "gender"] if c in meta_df.columns]
    meta_sub = meta_df.loc[:, cols_to_keep]

    # Conversion date -> meta_year
    if "date" in meta_sub.columns:
        meta_sub["meta_year"] = pd.to_numeric(meta_sub["date"], errors="coerce")
    else:
        meta_sub["meta_year"] = np.nan

    # Canon -> meta_canon (booléen 0/1)
    if "canon" in meta_sub.columns:
        meta_sub["meta_canon"] = (meta_sub["canon"] == "canon").astype(float)
    else:
        meta_sub["meta_canon"] = np.nan

    # Dummies pour le genre littéraire
    if "genre" in meta_sub.columns:
        genre_dummies = pd.get_dummies(meta_sub["genre"], prefix="meta_genre")
    else:
        genre_dummies = pd.DataFrame(index=meta_sub.index)

    # On conserve aussi la colonne 'gender' telle quelle (cible pour la prédiction)
    meta_features = pd.concat(
        [
            meta_sub[["gender", "meta_year", "meta_canon"]],
            genre_dummies,
        ],
        axis=1
    )

    # Jointure sur l'index (book_id)
    features_full = features_df.join(meta_features, how="left")
    return features_full


# -------------------------------------------------------------------
# Programme principal
# -------------------------------------------------------------------

def main():
    # 1. Parcours des .book
    book_paths = sorted(BOOK_DIR.glob("*.book"))
    rows = []

    print("Création des features de personnages pour : ")
    for path in book_paths:
        book_id = path.stem
        feats = extract_features_for_book(book_id, path)
        rows.append(feats)
        print(f" - {book_id}")

    if not rows:
        print("Aucun fichier .book trouvé, rien à faire.")
        return

    df = pd.DataFrame(rows).set_index("book_id")
    print(f"[INFO] Features de personnages : shape {df.shape}")

    # 2. Ajout des métadonnées (dont le genre de l'auteur)
    if META_PATH.exists():
        meta_df = pd.read_csv(META_PATH)
        df = add_metadata_features(df, meta_df)
        print(f"[INFO] Features + metadata : shape {df.shape}")
    else:
        print(f"[WARN] Fichier métadonnées introuvable : {META_PATH}")

    # 3. Suppression des colonnes avec >= 99 % de valeurs manquantes
    missing_ratio = df.isna().mean()
    keep_cols = missing_ratio < 0.99
    dropped_cols = missing_ratio[~keep_cols].index.tolist()

    df_filtered = df.loc[:, keep_cols]
    print(f"[INFO] Nombre de colonnes conservées : {df_filtered.shape[1]}")
    print(f"[INFO] Colonnes supprimées (>= 99 % NaN) : {len(dropped_cols)}")

    # 4. Sauvegarde
    df_filtered.to_csv(OUTPUT_PATH)
    print(f"[INFO] Fichier de features écrit dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
