#!/usr/bin/env python3
"""
Script pour filtrer un dataset HDF5 contenant des données LiDAR.
- Garder seulement certains champs
- Supprimer les points avec distance_cm == 0
- Convertir les RGB en classes (0, 1, 2, 3)
"""

import h5py
import numpy as np
import os
import argparse
from datetime import datetime

# Mapping RGB → classe
RGB_TO_CLASS = {
    (38, 23, 180): 0,   # Antenna
    (177, 132, 47): 1,   # Cable
    (129, 81, 97): 2,    # Electric pole
    (66, 132, 9): 3,     # Wind turbine
}

# Mapping inverse pour référence
CLASS_TO_RGB = {
    0: (38, 23, 180),
    1: (177, 132, 47),
    2: (129, 81, 97),
    3: (66, 132, 9),
}

def rgb_to_class(r, g, b):
    """Convertit un triplet RGB en ID de classe sans overflow."""
    # Convertir en int pour éviter les problèmes d'overflow
    r, g, b = int(r), int(g), int(b)
    
    # Tolérance de ±1 pour gérer les petites variations
    for (cr, cg, cb), class_id in RGB_TO_CLASS.items():
        if abs(r - cr) <= 1 and abs(g - cg) <= 1 and abs(b - cb) <= 1:
            return class_id
    return -1  # Point non classé (arrière-plan)

def rgb_to_class_vectorized(r_arr, g_arr, b_arr):
    """
    Version vectorisée plus rapide pour convertir les RGB en classes.
    Évite la boucle Python pour de meilleures performances.
    """
    # Convertir en int32 pour éviter les overflows
    r_arr = r_arr.astype(np.int32)
    g_arr = g_arr.astype(np.int32)
    b_arr = b_arr.astype(np.int32)
    
    # Initialiser avec -1 (non classé)
    class_ids = np.full(len(r_arr), -1, dtype=np.int8)
    
    # Pour chaque classe, créer un masque
    for (cr, cg, cb), class_id in RGB_TO_CLASS.items():
        # Tolérance de ±1 sur chaque canal
        mask = (np.abs(r_arr - cr) <= 1) & \
               (np.abs(g_arr - cg) <= 1) & \
               (np.abs(b_arr - cb) <= 1)
        class_ids[mask] = class_id
    
    return class_ids

def filtrer_dataset_h5(chemin_entree, chemin_sortie=None, keep_unlabeled=False, use_vectorized=True, verbose=True):
    """
    Filtre un fichier HDF5 pour garder uniquement certains champs,
    supprimer les points avec distance_cm == 0,
    et convertir les RGB en classes.
    
    Args:
        use_vectorized: Utiliser la version vectorisée (plus rapide) ou non
    """
    
    # Champs à garder (on remplace r,g,b par class_id)
    champs_a_garder = [
        ('distance_cm', '<u2'),
        ('azimuth_raw', '<i2'),
        ('elevation_raw', '<i2'),
        ('reflectivity', '<u2'),
        ('class_id', 'i1'),  # -1 à 3 (signé 8-bit)
        ('ego_x', '<f4'),    # Position du véhicule (en mètres)
        ('ego_y', '<f4'),
        ('ego_z', '<f4'),
        ('ego_yaw', '<f4'),  # en degrés
    ]
    
    # Générer le nom de sortie si non spécifié
    if chemin_sortie is None:
        base, ext = os.path.splitext(chemin_entree)
        chemin_sortie = f"{base}_classe{ext}"
    
    try:
        with h5py.File(chemin_entree, 'r') as f_entree:
            if verbose:
                print(f"\n📂 Ouverture du fichier : {chemin_entree}")
            
            # Chercher le dataset principal
            dataset_principal = None
            nom_dataset = None
            
            def trouver_dataset(name, obj):
                nonlocal dataset_principal, nom_dataset
                if isinstance(obj, h5py.Dataset):
                    if obj.dtype.names is not None and len(obj.dtype.names) > 0:
                        dataset_principal = obj
                        nom_dataset = name
                        return True
                return None
            
            f_entree.visititems(trouver_dataset)
            
            if dataset_principal is None:
                print("❌ Aucun dataset structuré trouvé")
                return False
            
            # Charger les données
            if verbose:
                print(f"📊 Chargement des données...")
            
            donnees = dataset_principal[:]
            nb_points_initial = len(donnees)
            
            if verbose:
                print(f"   ✅ {nb_points_initial} points chargés")
                print(f"   📋 Champs disponibles : {donnees.dtype.names}")
            
            # Vérifier les champs nécessaires
            champs_requis = ['distance_cm', 'azimuth_raw', 'elevation_raw', 
                           'reflectivity', 'r', 'g', 'b', 'ego_x', 'ego_y', 'ego_z', 'ego_yaw']
            
            champs_disponibles = set(donnees.dtype.names)
            champs_manquants = [champ for champ in champs_requis 
                               if champ not in champs_disponibles]
            
            if champs_manquants:
                print(f"❌ Champs manquants : {champs_manquants}")
                return False
            
            # Filtrer les points avec distance_cm == 0
            if verbose:
                print("\n🔍 Filtrage des points distance=0...")
            
            masque_distance = donnees['distance_cm'] != 0
            donnees_filtrees = donnees[masque_distance]
            nb_points_apres = len(donnees_filtrees)
            
            if verbose:
                print(f"   📏 Points avec distance=0 : {nb_points_initial - nb_points_apres}")
                print(f"   ✅ Points conservés : {nb_points_apres}")
            
            if nb_points_apres == 0:
                print("⚠️  Aucun point conservé après filtrage !")
                return False
            
            # Créer le tableau final
            nouveau_dtype = np.dtype(champs_a_garder)
            donnees_finales = np.zeros(nb_points_apres, dtype=nouveau_dtype)
            
            # Copier les champs simples
            for champ in ['distance_cm', 'azimuth_raw', 'elevation_raw', 'reflectivity']:
                donnees_finales[champ] = donnees_filtrees[champ]
            
            # Copier les champs ego (en convertissant cm → m)
            donnees_finales['ego_x'] = donnees_filtrees['ego_x'] / 100.0
            donnees_finales['ego_y'] = donnees_filtrees['ego_y'] / 100.0
            donnees_finales['ego_z'] = donnees_filtrees['ego_z'] / 100.0
            donnees_finales['ego_yaw'] = donnees_filtrees['ego_yaw'] / 100.0  # en degrés
            
            # Convertir RGB en class_id
            if verbose:
                print("\n🔄 Conversion RGB → classes...")
            
            if use_vectorized:
                # Version vectorisée (plus rapide)
                class_ids = rgb_to_class_vectorized(
                    donnees_filtrees['r'],
                    donnees_filtrees['g'],
                    donnees_filtrees['b']
                )
            else:
                # Version boucle (plus lente mais plus lisible)
                class_ids = []
                total = len(donnees_filtrees)
                for i in range(total):
                    if verbose and i % 1000000 == 0:
                        print(f"   Progression : {i}/{total} ({i/total*100:.1f}%)")
                    
                    r, g, b = donnees_filtrees[i]['r'], donnees_filtrees[i]['g'], donnees_filtrees[i]['b']
                    class_id = rgb_to_class(r, g, b)
                    class_ids.append(class_id)
                
                class_ids = np.array(class_ids, dtype='i1')
            
            donnees_finales['class_id'] = class_ids
            
            # Statistiques sur les classes
            if verbose:
                points_classes = donnees_finales[donnees_finales['class_id'] >= 0]
                points_non_classes = donnees_finales[donnees_finales['class_id'] == -1]
                
                print(f"\n📊 Statistiques des classes :")
                print(f"   Points non classés (arrière-plan) : {len(points_non_classes)} ({len(points_non_classes)/len(donnees_finales)*100:.1f}%)")
                
                for class_id in range(4):
                    mask = donnees_finales['class_id'] == class_id
                    count = np.sum(mask)
                    if count > 0:
                        label = ["Antenna", "Cable", "Electric pole", "Wind turbine"][class_id]
                        pourcentage = count / len(donnees_finales) * 100
                        print(f"   Classe {class_id} ({label}) : {count} points ({pourcentage:.2f}%)")
                
                # Vérifier les couleurs non reconnues
                if len(points_non_classes) > 0:
                    # Échantillonner quelques points non classés pour voir leurs couleurs
                    echantillon = points_non_classes[:min(10, len(points_non_classes))]
                    print(f"\n🔍 Échantillon de couleurs non reconnues :")
                    for i, idx in enumerate(np.random.choice(len(points_non_classes), min(5, len(points_non_classes)), replace=False)):
                        r = donnees_filtrees[idx]['r']
                        g = donnees_filtrees[idx]['g']
                        b = donnees_filtrees[idx]['b']
                        print(f"   {i}: RGB({r}, {g}, {b})")
            
            # Optionnel: ne garder que les points classés
            if not keep_unlabeled:
                mask_classes = donnees_finales['class_id'] >= 0
                donnees_finales = donnees_finales[mask_classes]
                if verbose:
                    print(f"\n🔍 Suppression des points non classés")
                    print(f"   Points restants : {len(donnees_finales)}")
            
            # Sauvegarder
            if verbose:
                print(f"\n💾 Sauvegarde dans : {chemin_sortie}")
            
            with h5py.File(chemin_sortie, 'w') as f_sortie:
                # Utiliser des chunks pour optimiser la lecture/écriture
                chunks = (min(100000, len(donnees_finales)),)
                
                f_sortie.create_dataset(
                    'data', 
                    data=donnees_finales,
                    chunks=chunks,
                    compression='gzip',
                    compression_opts=9,
                    shuffle=True
                )
                
                # Métadonnées
                f_sortie.attrs['description'] = "Données LiDAR avec classes"
                f_sortie.attrs['source_file'] = os.path.basename(chemin_entree)
                f_sortie.attrs['date_traitement'] = datetime.now().isoformat()
                f_sortie.attrs['points_originaux'] = nb_points_initial
                f_sortie.attrs['points_conserves'] = len(donnees_filtrees)
                f_sortie.attrs['points_classes'] = len(donnees_finales) if not keep_unlabeled else np.sum(class_ids >= 0)
                f_sortie.attrs['class_mapping'] = str(CLASS_TO_RGB)
            
            # Statistiques finales
            if verbose:
                taille_entree = os.path.getsize(chemin_entree) / (1024**3)  # GB
                taille_sortie = os.path.getsize(chemin_sortie) / (1024**3)  # GB
                
                print("\n" + "="*60)
                print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
                print("="*60)
                print(f"📁 Fichier source : {os.path.basename(chemin_entree)}")
                print(f"📁 Fichier destination : {os.path.basename(chemin_sortie)}")
                print(f"📊 Points originaux : {nb_points_initial:,}")
                print(f"📊 Points après filtrage distance : {nb_points_apres:,}")
                print(f"📊 Points avec classe : {len(donnees_finales):,}")
                print(f"📉 Taille originale : {taille_entree:.2f} GB")
                print(f"💾 Taille finale : {taille_sortie:.2f} GB")
                print(f"📉 Compression : {(1 - taille_sortie/taille_entree)*100:.1f}%")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Erreur lors du traitement : {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Filtrer et convertir les RGB en classes")
    parser.add_argument('fichier_entree', help='Fichier HDF5 d\'entrée')
    parser.add_argument('-o', '--output', help='Fichier de sortie')
    parser.add_argument('--keep-unlabeled', action='store_true', 
                       help='Garder les points non classés (arrière-plan)')
    parser.add_argument('--no-vectorized', action='store_true',
                       help='Désactiver la version vectorisée (plus lent mais plus facile à déboguer)')
    parser.add_argument('--no-verbose', action='store_true', help='Désactiver les messages')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.fichier_entree):
        print(f"❌ Fichier {args.fichier_entree} inexistant")
        return
    
    filtrer_dataset_h5(
        args.fichier_entree,
        args.output,
        keep_unlabeled=args.keep_unlabeled,
        use_vectorized=not args.no_vectorized,
        verbose=not args.no_verbose
    )


if __name__ == "__main__":
    main()