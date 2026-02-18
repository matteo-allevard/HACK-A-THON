#!/usr/bin/env python3
"""
Algorithme de détection d'objets par clustering
À partir des données LiDAR avec classes
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import h5py
from collections import defaultdict
import argparse

class LidarObjectDetector:
    def __init__(self, eps=0.5, min_samples=10):
        """
        Args:
            eps: Distance maximale entre deux points pour être dans le même cluster (mètres)
            min_samples: Nombre minimum de points pour former un cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        
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
        
        return df
    
    def cluster_by_frame_and_class(self, df):
        """
        Regroupe les points par frame (ego pose) et par classe,
        puis applique DBSCAN pour identifier les objets individuels
        """
        # Identifier les frames uniques
        frames = df.groupby(['ego_x', 'ego_y', 'ego_z', 'ego_yaw']).size().reset_index()
        frames = frames[['ego_x', 'ego_y', 'ego_z', 'ego_yaw']].drop_duplicates()
        
        all_objects = []
        
        for _, frame in frames.iterrows():
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
                
                # DBSCAN pour séparer les objets
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
                
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                labels = clustering.labels_
                
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
        
        return pd.DataFrame(all_objects)
    
    def create_bounding_box(self, points, class_id):
        """
        Crée une bounding box orientée à partir d'un ensemble de points
        """
        # Centrer les points
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        if class_id == 1:  # Cable : orientation spéciale (long et fin)
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
            width = 2 * perp_distances.max()  # Diamètre approximatif
            height = width  # Cable approximativement circulaire
            
            # Yaw = angle de la direction principale dans le plan XY
            yaw = np.arctan2(main_direction[1], main_direction[0])
            
        else:  # Objets compacts : bbox alignée sur les axes
            min_bounds = points.min(axis=0)
            max_bounds = points.max(axis=0)
            
            width = max_bounds[0] - min_bounds[0]
            length = max_bounds[1] - min_bounds[1]
            height = max_bounds[2] - min_bounds[2]
            
            # Pour les objets compacts, on peut utiliser une bbox non orientée
            yaw = 0.0
            
            centroid = (min_bounds + max_bounds) / 2
        
        return {
            'bbox_center_x': centroid[0],
            'bbox_center_y': centroid[1],
            'bbox_center_z': centroid[2],
            'bbox_width': width if 'width' in locals() else (max_bounds[0] - min_bounds[0]),
            'bbox_length': length if 'length' in locals() else (max_bounds[1] - min_bounds[1]),
            'bbox_height': height if 'height' in locals() else (max_bounds[2] - min_bounds[2]),
            'bbox_yaw': yaw if 'yaw' in locals() else 0.0
        }
    
    def detect_objects(self, h5_file):
        """
        Pipeline complet de détection
        """
        print(f"📂 Chargement des données depuis {h5_file}...")
        df = self.load_data(h5_file)
        
        print(f"📊 {len(df)} points chargés")
        
        print("🔍 Clustering par frame et classe...")
        objects_df = self.cluster_by_frame_and_class(df)
        
        print(f"✅ {len(objects_df)} objets détectés")
        
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


def main():
    parser = argparse.ArgumentParser(description="Détection d'objets par clustering")
    parser.add_argument('--input', required=True, help='Fichier HDF5 d\'entrée (filtré avec classes)')
    parser.add_argument('--output', default='predictions.csv', help='Fichier CSV de sortie')
    parser.add_argument('--eps', type=float, default=0.5, help='Rayon de clustering (mètres)')
    parser.add_argument('--min-samples', type=int, default=10, help='Points minimum par cluster')
    
    args = parser.parse_args()
    
    # Créer le détecteur
    detector = LidarObjectDetector(eps=args.eps, min_samples=args.min_samples)
    
    # Détecter les objets
    objects = detector.detect_objects(args.input)
    
    # Sauvegarder les prédictions
    detector.save_predictions(objects, args.output)
    
    # Afficher quelques exemples
    print("\n📋 Aperçu des 5 premiers objets :")
    print(objects[['class_label', 'bbox_center_x', 'bbox_center_y', 'bbox_center_z', 
                   'bbox_width', 'bbox_length', 'bbox_height']].head().to_string())


if __name__ == "__main__":
    main()