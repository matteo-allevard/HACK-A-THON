#!/usr/bin/env python3
"""
Algorithme de détection d'objets par clustering
À partir des données LiDAR avec classes
Version avec gestion améliorée de la mémoire et suivi des temps
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import h5py
import warnings
from collections import defaultdict
import argparse
import time
from datetime import timedelta

class LidarObjectDetector:
    def __init__(self, eps=0.5, min_samples=10, batch_size=5000):
        """
        Args:
            eps: Distance maximale entre deux points pour être dans le même cluster (mètres)
            min_samples: Nombre minimum de points pour former un cluster
            batch_size: Taille des lots pour le traitement par lots
        """
        self.eps = eps
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.max_points_per_group = 10000  # Nombre max de points par groupe
        self.timing_stats = defaultdict(list)  # Pour stocker les temps de traitement
        
    def spherical_to_cartesian(self, distance_cm, azimuth_raw, elevation_raw):
        """Convertit les coordonnées sphériques en cartésiennes (en mètres)"""
        distance_m = distance_cm / 100.0
        azimuth_rad = np.radians(azimuth_raw / 100.0)
        elevation_rad = np.radians(elevation_raw / 100.0)
        
        x = distance_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = -distance_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance_m * np.sin(elevation_rad)
        
        return np.column_stack((x, y, z))
    
    def load_data(self, h5_file):
        """Charge les données depuis le fichier HDF5 filtré"""
        start_time = time.time()
        
        with h5py.File(h5_file, 'r') as f:
            data = f['data'][:]
        
        df = pd.DataFrame({
            'distance_cm': data['distance_cm'],
            'azimuth_raw': data['azimuth_raw'],
            'elevation_raw': data['elevation_raw'],
            'reflectivity': data['reflectivity'],
            'class_id': data['class_id'],
            'ego_x': data['ego_x'],
            'ego_y': data['ego_y'],
            'ego_z': data['ego_z'],
            'ego_yaw': data['ego_yaw']
        })
        
        # Convertir en coordonnées cartésiennes
        xyz = self.spherical_to_cartesian(
            df['distance_cm'].values,
            df['azimuth_raw'].values,
            df['elevation_raw'].values
        )
        
        df['x'] = xyz[:, 0]
        df['y'] = xyz[:, 1]
        df['z'] = xyz[:, 2]
        
        elapsed = time.time() - start_time
        self.timing_stats['load_data'].append(elapsed)
        print(f"   ⏱️  Chargement: {elapsed:.2f} secondes")
        
        return df
    
    def cluster_points_safe(self, coords, eps, min_samples):
        """
        Applique DBSCAN de manière sécurisée avec gestion des erreurs mémoire
        """
        start_time = time.time()
        
        if len(coords) < min_samples:
            elapsed = time.time() - start_time
            self.timing_stats['cluster_points'].append(elapsed)
            return np.full(len(coords), -1)
        
        # Si trop de points, utiliser une approche par lots
        if len(coords) > self.max_points_per_group:
            print(f"⚠️  Groupe très large ({len(coords)} points). Division en sous-groupes...")
            result = self._cluster_large_group(coords, eps, min_samples)
            elapsed = time.time() - start_time
            self.timing_stats['cluster_large_group'].append(elapsed)
            return result
        
        try:
            # Essayer DBSCAN avec l'algorithme 'ball_tree' pour une meilleure efficacité mémoire
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clustering = DBSCAN(eps=eps, 
                                   min_samples=min_samples,
                                   metric='euclidean',
                                   algorithm='ball_tree',
                                   n_jobs=1).fit(coords)
            
            elapsed = time.time() - start_time
            self.timing_stats['cluster_dbscan'].append(elapsed)
            return clustering.labels_
            
        except MemoryError:
            print(f"⚠️  Erreur mémoire avec {len(coords)} points. Réduction de la taille...")
            # En cas d'erreur, diviser en lots plus petits
            result = self._cluster_large_group(coords, eps, min_samples, force_split=True)
            elapsed = time.time() - start_time
            self.timing_stats['cluster_recovery'].append(elapsed)
            return result
    
    def _cluster_large_group(self, coords, eps, min_samples, force_split=False):
        """
        Traite un grand groupe de points par lots
        """
        start_time = time.time()
        n_points = len(coords)
        all_labels = np.full(n_points, -1)
        next_label = 0
        
        # Déterminer la taille des lots
        if force_split:
            batch_size = min(1000, n_points // 10)
        else:
            batch_size = self.batch_size
        
        # Traiter par lots
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_coords = coords[i:end_idx]
            
            if len(batch_coords) < min_samples:
                continue
            
            try:
                batch_labels = DBSCAN(eps=eps, 
                                     min_samples=min_samples,
                                     metric='euclidean',
                                     algorithm='ball_tree',
                                     n_jobs=1).fit(batch_coords).labels_
                
                # Assigner les labels en évitant les conflits
                valid_mask = batch_labels >= 0
                if valid_mask.any():
                    # Ajouter un offset pour éviter les conflits entre lots
                    all_labels[i:end_idx][valid_mask] = batch_labels[valid_mask] + next_label
                    if valid_mask.any():
                        next_label = all_labels.max() + 1
                        
            except MemoryError:
                print(f"  → Lot trop grand ({len(batch_coords)} points), division supplémentaire...")
                # Récursion pour diviser encore plus
                sub_labels = self._cluster_large_group(batch_coords, eps, min_samples, force_split=True)
                all_labels[i:end_idx] = sub_labels + next_label if sub_labels.max() >= 0 else sub_labels
                next_label = all_labels.max() + 1
        
        elapsed = time.time() - start_time
        self.timing_stats['cluster_large_batch'].append(elapsed)
        return all_labels
    
    def cluster_by_frame_and_class(self, df):
        """
        Regroupe les points par frame (ego pose) et par classe,
        puis applique DBSCAN pour identifier les objets individuels
        Version avec gestion de mémoire améliorée
        """
        total_start_time = time.time()
        
        # Identifier les frames uniques
        frames = df.groupby(['ego_x', 'ego_y', 'ego_z', 'ego_yaw']).size().reset_index()
        frames = frames[['ego_x', 'ego_y', 'ego_z', 'ego_yaw']].drop_duplicates()
        
        all_objects = []
        total_frames = len(frames)
        frame_times = []
        
        print(f"\n📊 Traitement de {total_frames} frames...")
        
        for idx, (_, frame) in enumerate(frames.iterrows()):
            frame_start = time.time()
            
            if idx % 10 == 0 and idx > 0:
                avg_time = np.mean(frame_times[-10:]) if frame_times else 0
                remaining = (total_frames - idx) * avg_time
                print(f"  Frame {idx}/{total_frames} (⏱️ moy: {avg_time:.2f}s, restant: {timedelta(seconds=int(remaining))})")
            
            # Filtrer les points de cette frame
            mask = (
                (df['ego_x'] == frame['ego_x']) &
                (df['ego_y'] == frame['ego_y']) &
                (df['ego_z'] == frame['ego_z']) &
                (df['ego_yaw'] == frame['ego_yaw'])
            )
            frame_points = df[mask].copy()
            
            if len(frame_points) == 0:
                continue
            
            # Pour chaque classe, faire du clustering séparé
            for class_id in range(4):  # 0,1,2,3
                class_points = frame_points[frame_points['class_id'] == class_id]
                
                if len(class_points) < self.min_samples:
                    continue
                
                # Extraire les coordonnées 3D
                coords = class_points[['x', 'y', 'z']].values
                
                # Paramètres adaptés selon la classe
                if class_id == 1:  # Cable : plus sensible car fin et long
                    eps = self.eps * 0.3
                    min_samples = max(3, self.min_samples // 3)
                elif class_id == 3:  # Wind turbine : grand
                    eps = self.eps * 2.0
                    min_samples = self.min_samples * 2
                else:
                    eps = self.eps
                    min_samples = self.min_samples
                
                # Clustering avec gestion mémoire
                labels = self.cluster_points_safe(coords, eps, min_samples)
                
                # Pour chaque cluster, créer un objet
                unique_labels = set(labels)
                for label in unique_labels:
                    if label == -1:  # Bruit
                        continue
                    
                    cluster_mask = labels == label
                    cluster_points = coords[cluster_mask]
                    
                    # Créer la bounding box
                    bbox = self.create_bounding_box(cluster_points, class_id)
                    
                    # Ajouter les informations de la frame
                    bbox.update({
                        'ego_x': frame['ego_x'],
                        'ego_y': frame['ego_y'],
                        'ego_z': frame['ego_z'],
                        'ego_yaw': frame['ego_yaw'],
                        'class_id': class_id,
                        'class_label': ['Antenna', 'Cable', 'Electric pole', 'Wind turbine'][class_id],
                        'num_points': len(cluster_points)
                    })
                    
                    all_objects.append(bbox)
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            self.timing_stats['per_frame'].append(frame_time)
        
        total_time = time.time() - total_start_time
        self.timing_stats['total_clustering'] = total_time
        
        # Afficher les statistiques de temps par frame
        if frame_times:
            print(f"\n⏱️  Statistiques de traitement par frame:")
            print(f"   Min: {min(frame_times):.2f}s")
            print(f"   Max: {max(frame_times):.2f}s")
            print(f"   Moy: {np.mean(frame_times):.2f}s")
            print(f"   Médiane: {np.median(frame_times):.2f}s")
            print(f"   Écart-type: {np.std(frame_times):.2f}s")
        
        return pd.DataFrame(all_objects)
    
    def create_bounding_box(self, points, class_id):
        """
        Crée une bounding box orientée à partir d'un ensemble de points
        """
        start_time = time.time()
        
        # Centrer les points
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        # Initialiser les variables
        width = length = height = yaw = 0.0
        min_bounds = None
        max_bounds = None
        
        if class_id == 1:  # Cable : orientation spéciale (long et fin)
            try:
                # PCA pour trouver la direction principale
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # La direction principale est celle avec la plus grande valeur propre
                main_direction = eigenvectors[:, -1]
                
                # Projeter les points sur cette direction
                projections = centered @ main_direction
                proj_min, proj_max = projections.min(), projections.max()
                
                # Calculer les dimensions perpendiculaires
                perp_points = centered - np.outer(projections, main_direction)
                perp_distances = np.linalg.norm(perp_points, axis=1)
                
                length = proj_max - proj_min
                width = 2 * perp_distances.max() if len(perp_distances) > 0 else 0.1
                height = width  # Cable approximativement circulaire
                
                # Yaw = angle de la direction principale dans le plan XY
                yaw = np.arctan2(main_direction[1], main_direction[0])
                
            except np.linalg.LinAlgError:
                # En cas d'erreur PCA, utiliser bbox simple
                min_bounds = points.min(axis=0)
                max_bounds = points.max(axis=0)
                width = max_bounds[0] - min_bounds[0]
                length = max_bounds[1] - min_bounds[1]
                height = max_bounds[2] - min_bounds[2]
                yaw = 0.0
                
        else:  # Objets compacts : bbox alignée sur les axes
            min_bounds = points.min(axis=0)
            max_bounds = points.max(axis=0)
            
            width = max_bounds[0] - min_bounds[0]
            length = max_bounds[1] - min_bounds[1]
            height = max_bounds[2] - min_bounds[2]
            
            # Pour les objets compacts, on peut utiliser une bbox non orientée
            yaw = 0.0
            
            centroid = (min_bounds + max_bounds) / 2
        
        # S'assurer que les dimensions sont positives
        width = max(width, 0.1)
        length = max(length, 0.1)
        height = max(height, 0.1)
        
        elapsed = time.time() - start_time
        self.timing_stats['create_bbox'].append(elapsed)
        
        return {
            'bbox_center_x': centroid[0],
            'bbox_center_y': centroid[1],
            'bbox_center_z': centroid[2],
            'bbox_width': width,
            'bbox_length': length,
            'bbox_height': height,
            'bbox_yaw': yaw
        }
    
    def detect_objects(self, h5_file):
        """
        Pipeline complet de détection
        """
        pipeline_start = time.time()
        
        print(f"📂 Chargement des données depuis {h5_file}...")
        df = self.load_data(h5_file)
        
        print(f"📊 {len(df)} points chargés")
        print(f"   Classes: {df['class_id'].value_counts().to_dict()}")
        
        print("🔍 Clustering par frame et classe...")
        print("   (avec gestion améliorée de la mémoire)")
        
        objects_df = self.cluster_by_frame_and_class(df)
        
        total_time = time.time() - pipeline_start
        self.timing_stats['total_pipeline'] = total_time
        
        print(f"\n✅ {len(objects_df)} objets détectés")
        
        # Statistiques par classe
        print("\n📊 Statistiques par classe :")
        for class_id in range(4):
            count = len(objects_df[objects_df['class_id'] == class_id])
            if count > 0:
                label = ['Antenna', 'Cable', 'Electric pole', 'Wind turbine'][class_id]
                print(f"   {label}: {count} objets")
        
        return objects_df
    
    def save_predictions(self, objects_df, output_file):
        """
        Sauvegarde au format CSV requis pour la compétition
        """
        if objects_df.empty:
            print("⚠️  Aucun objet à sauvegarder")
            # Créer un DataFrame vide avec les bonnes colonnes
            objects_df = pd.DataFrame(columns=[
                'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
                'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
                'bbox_width', 'bbox_length', 'bbox_height',
                'bbox_yaw', 'class_id', 'class_label'
            ])
        
        # Sélectionner et ordonner les colonnes
        columns = [
            'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
            'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
            'bbox_width', 'bbox_length', 'bbox_height',
            'bbox_yaw', 'class_id', 'class_label'
        ]
        
        # S'assurer que toutes les colonnes existent
        for col in columns:
            if col not in objects_df.columns:
                if col == 'class_label':
                    objects_df[col] = objects_df['class_id'].map({
                        0: 'Antenna', 1: 'Cable', 2: 'Electric pole', 3: 'Wind turbine'
                    })
                else:
                    objects_df[col] = 0.0
        
        objects_df[columns].to_csv(output_file, index=False)
        print(f"💾 Prédictions sauvegardées dans {output_file}")
    
    def print_timing_summary(self):
        """
        Affiche un résumé détaillé des temps de traitement
        """
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DES TEMPS DE TRAITEMENT")
        print("="*60)
        
        if 'load_data' in self.timing_stats:
            print(f"\n📂 Chargement des données:")
            print(f"   Total: {sum(self.timing_stats['load_data']):.2f}s")
        
        if 'total_clustering' in self.timing_stats:
            print(f"\n🔍 Clustering:")
            print(f"   Total: {self.timing_stats['total_clustering']:.2f}s")
        
        if 'per_frame' in self.timing_stats:
            frame_times = self.timing_stats['per_frame']
            print(f"\n📊 Par frame ({len(frame_times)} frames):")
            print(f"   Min: {min(frame_times):.2f}s")
            print(f"   Max: {max(frame_times):.2f}s")
            print(f"   Moy: {np.mean(frame_times):.2f}s")
            print(f"   Médiane: {np.median(frame_times):.2f}s")
            print(f"   Total: {sum(frame_times):.2f}s")
        
        if 'cluster_dbscan' in self.timing_stats:
            dbscan_times = self.timing_stats['cluster_dbscan']
            print(f"\n⚡ DBSCAN (normal):")
            print(f"   {len(dbscan_times)} appels")
            print(f"   Moy: {np.mean(dbscan_times):.3f}s")
            print(f"   Total: {sum(dbscan_times):.2f}s")
        
        if 'cluster_large_group' in self.timing_stats:
            large_times = self.timing_stats['cluster_large_group']
            print(f"\n🔄 Clustering grands groupes:")
            print(f"   {len(large_times)} appels")
            print(f"   Moy: {np.mean(large_times):.2f}s")
            print(f"   Total: {sum(large_times):.2f}s")
        
        if 'create_bbox' in self.timing_stats:
            bbox_times = self.timing_stats['create_bbox']
            print(f"\n📦 Création bounding boxes:")
            print(f"   {len(bbox_times)} boxes")
            print(f"   Moy: {np.mean(bbox_times)*1000:.2f}ms")
            print(f"   Total: {sum(bbox_times):.2f}s")
        
        if 'total_pipeline' in self.timing_stats:
            print(f"\n⏱️  TEMPS TOTAL D'EXÉCUTION: {self.timing_stats['total_pipeline']:.2f}s")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Détection d'objets par clustering")
    parser.add_argument('--input', required=True, help='Fichier HDF5 d\'entrée (filtré avec classes)')
    parser.add_argument('--output', default='predictions.csv', help='Fichier CSV de sortie')
    parser.add_argument('--eps', type=float, default=0.5, help='Rayon de clustering (mètres)')
    parser.add_argument('--min-samples', type=int, default=10, help='Points minimum par cluster')
    parser.add_argument('--batch-size', type=int, default=5000, help='Taille des lots pour le clustering')
    parser.add_argument('--timing', action='store_true', help='Afficher le résumé détaillé des temps')
    
    args = parser.parse_args()
    
    # Créer le détecteur avec gestion mémoire
    detector = LidarObjectDetector(
        eps=args.eps, 
        min_samples=args.min_samples,
        batch_size=args.batch_size
    )
    
    try:
        # Détecter les objets
        objects = detector.detect_objects(args.input)
        
        # Sauvegarder les prédictions
        detector.save_predictions(objects, args.output)
        
        # Afficher quelques exemples
        if not objects.empty:
            print("\n📋 Aperçu des 5 premiers objets :")
            display_cols = ['class_label', 'bbox_center_x', 'bbox_center_y', 'bbox_center_z', 
                           'bbox_width', 'bbox_length', 'bbox_height', 'num_points']
            available_cols = [col for col in display_cols if col in objects.columns]
            print(objects[available_cols].head().to_string())
        
        # Afficher le résumé des temps si demandé
        if args.timing:
            detector.print_timing_summary()
        
    except MemoryError as e:
        print(f"❌ Erreur mémoire fatale: {e}")
        print("   Suggestions:")
        print("   - Réduire --batch-size (essayez 2000)")
        print("   - Augmenter --min-samples pour réduire le nombre de clusters")
        print("   - Utiliser --eps plus grand pour moins de clusters")
        raise
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        raise


if __name__ == "__main__":
    main()