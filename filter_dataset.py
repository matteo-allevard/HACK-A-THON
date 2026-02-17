#!/usr/bin/env python3
"""
Script pour filtrer un dataset HDF5 contenant des données LiDAR/caméra.
Garder seulement certains champs et supprimer les points avec distance_cm == 0.
"""

import h5py
import numpy as np
import os
import argparse
from datetime import datetime

def filtrer_dataset_h5(chemin_entree, chemin_sortie=None, verbose=True):
    """
    Filtre un fichier HDF5 pour garder uniquement certains champs
    et supprimer les points avec distance_cm == 0.
    
    Args:
        chemin_entree (str): Chemin vers le fichier HDF5 d'entrée
        chemin_sortie (str, optional): Chemin pour le fichier de sortie.
                                      Si None, ajoute "_filtre" au nom original.
        verbose (bool): Afficher des informations détaillées
    
    Returns:
        bool: True si le traitement a réussi, False sinon
    """
    
    # Champs à garder
    champs_a_garder = [
        ('distance_cm', '<u2'),
        ('azimuth_raw', '<i2'),
        ('elevation_raw', '<i2'),
        ('reflectivity', '<u2'),
        ('r', 'u1'),
        ('g', 'u1'),
        ('b', 'u1')
    ]
    
    # Générer le nom de sortie si non spécifié
    if chemin_sortie is None:
        base, ext = os.path.splitext(chemin_entree)
        chemin_sortie = f"{base}_filtre{ext}"
    
    try:
        # Ouvrir le fichier d'entrée
        if verbose:
            print(f"\n📂 Ouverture du fichier : {chemin_entree}")
            print(f"⏱️  {datetime.now().strftime('%H:%M:%S')}")
        
        with h5py.File(chemin_entree, 'r') as f_entree:
            # Vérifier la structure du fichier
            if verbose:
                print("\n📋 Structure du fichier HDF5 :")
                f_entree.visititems(lambda name, obj: 
                    print(f"   - {name}: {type(obj).__name__}") if isinstance(obj, h5py.Dataset) else None)
            
            # Chercher le dataset principal
            dataset_principal = None
            nom_dataset = None
            
            def trouver_dataset(name, obj):
                nonlocal dataset_principal, nom_dataset
                if isinstance(obj, h5py.Dataset):
                    # Vérifier si ce dataset a la structure attendue
                    if obj.dtype.names is not None and len(obj.dtype.names) > 0:
                        dataset_principal = obj
                        nom_dataset = name
                        return True  # Arrêter la recherche
                return None
            
            f_entree.visititems(trouver_dataset)
            
            if dataset_principal is None:
                print("❌ Aucun dataset structuré trouvé dans le fichier")
                return False
            
            if verbose:
                print(f"\n📊 Dataset trouvé : '{nom_dataset}'")
                print(f"   Shape: {dataset_principal.shape}")
                print(f"   Types: {dataset_principal.dtype}")
                print(f"   Champs disponibles: {dataset_principal.dtype.names}")
            
            # Charger les données
            if verbose:
                print("\n🔄 Chargement des données...")
            
            donnees = dataset_principal[:]
            nb_points_initial = len(donnees)
            
            if verbose:
                print(f"   ✅ {nb_points_initial} points chargés")
            
            # Vérifier que tous les champs à garder existent
            champs_disponibles = set(donnees.dtype.names)
            champs_manquants = [champ[0] for champ in champs_a_garder 
                               if champ[0] not in champs_disponibles]
            
            if champs_manquants:
                print(f"❌ Champs manquants dans le dataset : {champs_manquants}")
                print(f"   Champs disponibles : {champs_disponibles}")
                return False
            
            # Filtrer les points avec distance_cm == 0
            if verbose:
                print("\n🔍 Filtrage des points...")
            
            masque_distance = donnees['distance_cm'] != 0
            donnees_filtrees = donnees[masque_distance]
            nb_points_apres = len(donnees_filtrees)
            
            if verbose:
                print(f"   📏 Points avec distance=0 : {nb_points_initial - nb_points_apres}")
                print(f"   ✅ Points conservés : {nb_points_apres}")
            
            if nb_points_apres == 0:
                print("⚠️  Aucun point conservé après filtrage !")
                return False
            
            # Créer un nouveau tableau avec seulement les champs souhaités
            if verbose:
                print("\n🔄 Création du nouveau tableau avec les champs sélectionnés...")
            
            # Préparer la structure pour le nouveau dtype
            nouveau_dtype = np.dtype(champs_a_garder)
            donnees_finales = np.zeros(nb_points_apres, dtype=nouveau_dtype)
            
            # Copier les données pour chaque champ
            for champ, _ in champs_a_garder:
                donnees_finales[champ] = donnees_filtrees[champ]
            
            if verbose:
                print(f"   ✅ Nouveau dtype : {nouveau_dtype}")
            
            # Sauvegarder dans un nouveau fichier
            if verbose:
                print(f"\n💾 Sauvegarde dans : {chemin_sortie}")
            
            with h5py.File(chemin_sortie, 'w') as f_sortie:
                # Créer le dataset avec compression pour économiser de l'espace
                f_sortie.create_dataset(
                    'data', 
                    data=donnees_finales,
                    compression='gzip',
                    compression_opts=9,  # Compression maximale
                    shuffle=True          # Améliore la compression
                )
                
                # Ajouter des métadonnées
                f_sortie.attrs['description'] = "Données LiDAR filtrées"
                f_sortie.attrs['source_file'] = os.path.basename(chemin_entree)
                f_sortie.attrs['date_traitement'] = datetime.now().isoformat()
                f_sortie.attrs['points_originaux'] = nb_points_initial
                f_sortie.attrs['points_conserves'] = nb_points_apres
                f_sortie.attrs['champs_conserves'] = str([champ[0] for champ in champs_a_garder])
            
            # Afficher le résumé
            if verbose:
                print("\n" + "="*50)
                print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
                print("="*50)
                print(f"📁 Fichier source : {os.path.basename(chemin_entree)}")
                print(f"📁 Fichier destination : {os.path.basename(chemin_sortie)}")
                print(f"📊 Points originaux : {nb_points_initial}")
                print(f"📊 Points après filtrage : {nb_points_apres}")
                print(f"📉 Réduction : {((nb_points_initial - nb_points_apres) / nb_points_initial * 100):.1f}%")
                
                # Statistiques sur les distances
                if nb_points_apres > 0:
                    distances = donnees_finales['distance_cm']
                    print(f"\n📏 Statistiques des distances conservées :")
                    print(f"   Min : {distances.min()} cm")
                    print(f"   Max : {distances.max()} cm")
                    print(f"   Moyenne : {distances.mean():.1f} cm")
                    print(f"   Médiane : {np.median(distances):.1f} cm")
                
                # Taille des fichiers
                taille_entree = os.path.getsize(chemin_entree) / (1024*1024)
                taille_sortie = os.path.getsize(chemin_sortie) / (1024*1024)
                print(f"\n💾 Taille du fichier original : {taille_entree:.2f} MB")
                print(f"💾 Taille du fichier filtré : {taille_sortie:.2f} MB")
                if taille_entree > 0:
                    print(f"📉 Compression : {(1 - taille_sortie/taille_entree)*100:.1f}%")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Erreur lors du traitement : {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verifier_fichier_sortie(chemin_fichier):
    """
    Vérifie le contenu du fichier de sortie.
    """
    try:
        with h5py.File(chemin_fichier, 'r') as f:
            print(f"\n🔍 Vérification du fichier de sortie :")
            print(f"   Dataset: {list(f.keys())}")
            
            if 'data' in f:
                data = f['data']
                print(f"   Shape: {data.shape}")
                print(f"   Type: {data.dtype}")
                print(f"   Champs: {data.dtype.names}")
                print(f"   Attributs: {dict(f.attrs)}")
                
                # Afficher les 5 premiers points
                print(f"\n📋 Aperçu des 5 premiers points :")
                for i in range(min(5, len(data))):
                    point = data[i]
                    print(f"   {i}: ", end="")
                    for champ in data.dtype.names:
                        print(f"{champ}={point[champ]} ", end="")
                    print()
                
                return True
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Filtrer un fichier HDF5 de données LiDAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s data.h5
  %(prog)s data.h5 -o data_filtre.h5
  %(prog)s data.h5 --no-verbose
        """
    )
    
    parser.add_argument(
        'fichier_entree',
        help='Chemin vers le fichier HDF5 d\'entrée'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Chemin pour le fichier de sortie (défaut: [nom]_filtre.h5)'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Désactiver les messages détaillés'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Vérifier le fichier de sortie après traitement'
    )
    
    args = parser.parse_args()
    
    # Vérifier que le fichier d'entrée existe
    if not os.path.exists(args.fichier_entree):
        print(f"❌ Le fichier {args.fichier_entree} n'existe pas !")
        return
    
    # Exécuter le filtrage
    success = filtrer_dataset_h5(
        args.fichier_entree,
        args.output,
        verbose=not args.no_verbose
    )
    
    # Vérifier le résultat si demandé
    if success and args.verify and args.output:
        verifier_fichier_sortie(args.output)
    elif success and args.verify and not args.output:
        # Si pas de output spécifié, utiliser le nom par défaut
        base, ext = os.path.splitext(args.fichier_entree)
        output_defaut = f"{base}_filtre{ext}"
        verifier_fichier_sortie(output_defaut)


if __name__ == "__main__":
    main()