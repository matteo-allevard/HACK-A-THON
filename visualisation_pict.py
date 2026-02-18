#!/usr/bin/env python3
"""
Visualisation 3D des points LiDAR et des bounding boxes détectées
Version compatible avec les fichiers originaux et filtrés
"""

import numpy as np
import pandas as pd
import open3d as o3d
import h5py
import argparse
import os

class LidarVisualizer:
    def __init__(self):
        # Couleurs pour les classes (RGB normalisé 0-1)
        self.class_colors = {
            0: [0.15, 0.09, 0.71],  # Antenna - bleu
            1: [0.69, 0.52, 0.18],  # Cable - marron/orange
            2: [0.51, 0.32, 0.38],  # Electric pole - violet
            3: [0.26, 0.52, 0.04],  # Wind turbine - vert
            -1: [0.5, 0.5, 0.5]     # Arrière-plan - gris
        }
        
        # Couleurs pour les bounding boxes (plus vives)
        self.bbox_colors = {
            0: [0, 0, 1],     # Bleu vif
            1: [1, 0.5, 0],   # Orange
            2: [0.8, 0, 0.8], # Violet
            3: [0, 1, 0]      # Vert vif
        }
    
    def load_data(self, h5_file):
        """Charge les points depuis le fichier HDF5 (original ou filtré)"""
        print(f"📂 Chargement des points depuis {h5_file}...")
        
        with h5py.File(h5_file, 'r') as f:
            # Chercher le dataset approprié
            if 'data' in f:
                # Format filtré
                print("   Format: filtré (dataset 'data')")
                data = f['data'][:]
                
                df = pd.DataFrame({
                    'distance_cm': data['distance_cm'],
                    'azimuth_raw': data['azimuth_raw'],
                    'elevation_raw': data['elevation_raw'],
                    'class_id': data['class_id'],
                })
                
                # Ajouter ego_x, ego_y, ego_z, ego_yaw si présents
                if 'ego_x' in data.dtype.names:
                    df['ego_x'] = data['ego_x']
                    df['ego_y'] = data['ego_y']
                    df['ego_z'] = data['ego_z']
                    df['ego_yaw'] = data['ego_yaw']
                
            elif 'lidar_points' in f:
                # Format original
                print("   Format: original (dataset 'lidar_points')")
                data = f['lidar_points'][:]
                
                df = pd.DataFrame({
                    'distance_cm': data['distance_cm'],
                    'azimuth_raw': data['azimuth_raw'],
                    'elevation_raw': data['elevation_raw'],
                    'reflectivity': data['reflectivity'],
                })
                
                # Ajouter les champs ego
                df['ego_x'] = data['ego_x'] / 100.0  # cm -> m
                df['ego_y'] = data['ego_y'] / 100.0
                df['ego_z'] = data['ego_z'] / 100.0
                df['ego_yaw'] = data['ego_yaw'] / 100.0  # 1/100 deg -> deg
                
                # Classer les points par couleur RGB
                print("   🔍 Classification des points par couleur...")
                df['class_id'] = -1  # Default: arrière-plan
                
                # Mapping RGB -> classe
                rgb_to_class = {
                    (38, 23, 180): 0,   # Antenna
                    (177, 132, 47): 1,   # Cable
                    (129, 81, 97): 2,    # Electric pole
                    (66, 132, 9): 3,     # Wind turbine
                }
                
                # Version vectorisée pour la classification
                r = data['r'].astype(np.int32)
                g = data['g'].astype(np.int32)
                b = data['b'].astype(np.int32)
                
                for (cr, cg, cb), class_id in rgb_to_class.items():
                    mask = (np.abs(r - cr) <= 1) & (np.abs(g - cg) <= 1) & (np.abs(b - cb) <= 1)
                    df.loc[mask, 'class_id'] = class_id
                
            else:
                # Chercher n'importe quel dataset avec des points
                found = False
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) > 0:
                        if f[key].dtype.names is not None:
                            print(f"   Dataset trouvé: '{key}'")
                            data = f[key][:]
                            found = True
                            break
                
                if not found:
                    raise ValueError("Aucun dataset de points trouvé dans le fichier")
                
                # Créer DataFrame basique
                df = pd.DataFrame({name: data[name] for name in data.dtype.names})
        
        # Convertir en coordonnées cartésiennes
        print("🔄 Calcul des coordonnées cartésiennes...")
        distance_m = df['distance_cm'] / 100.0
        azimuth_rad = np.radians(df['azimuth_raw'] / 100.0)
        elevation_rad = np.radians(df['elevation_raw'] / 100.0)
        
        df['x'] = distance_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
        df['y'] = -distance_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
        df['z'] = distance_m * np.sin(elevation_rad)
        
        print(f"   ✅ {len(df)} points chargés")
        
        # Statistiques des classes
        if 'class_id' in df.columns:
            n_classified = len(df[df['class_id'] >= 0])
            print(f"   📊 Points classés: {n_classified}/{len(df)} ({n_classified/len(df)*100:.1f}%)")
        
        return df
    
    def load_predictions(self, csv_file):
        """Charge les prédictions depuis le CSV"""
        if not os.path.exists(csv_file):
            print(f"⚠️  Fichier CSV non trouvé: {csv_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_file)
        print(f"📂 {len(df)} prédictions chargées")
        return df
    
    def create_oriented_bbox(self, center, width, length, height, yaw, color):
        """
        Crée une boîte 3D orientée pour visualisation
        """
        # Éviter les boîtes de taille zéro
        if width <= 0 or length <= 0 or height <= 0:
            return None
        
        # Créer une boîte orientée
        bbox = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zyx([yaw, 0, 0]),
            extent=[width, length, height]
        )
        bbox.color = color
        return bbox
    
    def create_bbox_lines(self, center, width, length, height, yaw, color):
        """
        Crée une boîte en mode fil de fer (wireframe)
        """
        if width <= 0 or length <= 0 or height <= 0:
            return None
        
        # Points de la boîte non orientée
        corners = np.array([
            [-width/2, -length/2, -height/2],
            [width/2, -length/2, -height/2],
            [width/2, length/2, -height/2],
            [-width/2, length/2, -height/2],
            [-width/2, -length/2, height/2],
            [width/2, -length/2, height/2],
            [width/2, length/2, height/2],
            [-width/2, length/2, height/2]
        ])
        
        # Appliquer rotation
        if yaw != 0:
            rot_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            corners = corners @ rot_matrix.T
        
        # Ajouter translation
        corners += center
        
        # Définir les lignes (arêtes de la boîte)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Base
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Verticales
        ]
        
        # Créer le line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Couleur des lignes
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def filter_by_frame(self, df_points, df_predictions, frame_idx=0):
        """
        Filtre les points et prédictions pour une frame spécifique
        """
        if len(df_predictions) == 0:
            return df_points, df_predictions
        
        # Vérifier que les colonnes ego existent
        if not all(col in df_points.columns for col in ['ego_x', 'ego_y', 'ego_z', 'ego_yaw']):
            print("⚠️  Pas d'information de frame dans les points - affichage de tous les points")
            return df_points, df_predictions
        
        # Prendre la première frame unique des prédictions
        unique_frames = df_predictions[['ego_x', 'ego_y', 'ego_z', 'ego_yaw']].drop_duplicates()
        
        if len(unique_frames) == 0:
            return df_points, df_predictions
        
        if frame_idx >= len(unique_frames):
            print(f"⚠️  Frame {frame_idx} hors limites, utilisation de la frame 0")
            frame_idx = 0
        
        frame = unique_frames.iloc[frame_idx]
        
        # Filtrer les points
        mask_points = (
            (np.abs(df_points['ego_x'] - frame['ego_x']) < 0.01) &
            (np.abs(df_points['ego_y'] - frame['ego_y']) < 0.01) &
            (np.abs(df_points['ego_z'] - frame['ego_z']) < 0.01) &
            (np.abs(df_points['ego_yaw'] - frame['ego_yaw']) < 0.01)
        )
        
        # Filtrer les prédictions
        mask_preds = (
            (np.abs(df_predictions['ego_x'] - frame['ego_x']) < 0.01) &
            (np.abs(df_predictions['ego_y'] - frame['ego_y']) < 0.01) &
            (np.abs(df_predictions['ego_z'] - frame['ego_z']) < 0.01) &
            (np.abs(df_predictions['ego_yaw'] - frame['ego_yaw']) < 0.01)
        )
        
        return df_points[mask_points], df_predictions[mask_preds]
    
    def visualize(self, h5_file, csv_file, frame_idx=0, point_size=2.0, window_size=(1280, 720)):
        """
        Visualise les points et les bounding boxes
        """
        # Charger les points
        df_points = self.load_data(h5_file)
        
        # Charger les prédictions
        df_preds = self.load_predictions(csv_file)
        
        # Filtrer par frame si possible
        if len(df_preds) > 0:
            df_points, df_preds = self.filter_by_frame(df_points, df_preds, frame_idx)
        
        print(f"\n📊 Frame {frame_idx}:")
        print(f"   Points: {len(df_points)}")
        print(f"   Objets détectés: {len(df_preds)}")
        
        # Créer le nuage de points
        pcd = o3d.geometry.PointCloud()
        points = df_points[['x', 'y', 'z']].values
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Colorier les points par classe
        if 'class_id' in df_points.columns:
            colors = np.zeros((len(df_points), 3))
            for class_id, color in self.class_colors.items():
                mask = df_points['class_id'] == class_id
                colors[mask] = color
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Créer les bounding boxes
        geometries = [pcd]
        
        if len(df_preds) > 0:
            print("\n📋 Objets dans cette frame:")
            for _, pred in df_preds.iterrows():
                center = [pred['bbox_center_x'], pred['bbox_center_y'], pred['bbox_center_z']]
                
                # Afficher les infos de l'objet
                size_info = f"{pred['bbox_width']:.1f}, {pred['bbox_length']:.1f}, {pred['bbox_height']:.1f}"
                print(f"   {pred['class_label']}: pos=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) size=({size_info})")
                
                # Créer la boîte si elle a une taille valide
                if pred['bbox_width'] > 0.01 and pred['bbox_length'] > 0.01 and pred['bbox_height'] > 0.01:
                    bbox = self.create_oriented_bbox(
                        center=center,
                        width=pred['bbox_width'],
                        length=pred['bbox_length'],
                        height=pred['bbox_height'],
                        yaw=pred['bbox_yaw'],
                        color=self.bbox_colors.get(pred['class_id'], [1, 0, 0])
                    )
                    if bbox is not None:
                        geometries.append(bbox)
        
        # Statistiques de la scène
        print(f"\n📊 Statistiques de la scène:")
        print(f"   X: [{points[:,0].min():.1f}, {points[:,0].max():.1f}]")
        print(f"   Y: [{points[:,1].min():.1f}, {points[:,1].max():.1f}]")
        print(f"   Z: [{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
        
        # Visualisation
        print("\n🖼️  Lancement de la visualisation...")
        print("   Contrôles: Souris = rotation, Molette = zoom, Ctrl + souris = déplacement")
        print("   (Les warnings OpenGL peuvent être ignorés si la fenêtre s'ouvre)")
        
        try:
            # Configurer la visualisation
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Lidar Visualization - Frame {frame_idx}", 
                            width=window_size[0], height=window_size[1])
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            # Options de rendu
            opt = vis.get_render_option()
            opt.point_size = point_size
            opt.background_color = np.array([0.1, 0.1, 0.1])  # Gris foncé
            
            # Configuration de la caméra
            ctrl = vis.get_view_control()
            ctrl.set_front([0, -1, 0])  # Vue de côté
            ctrl.set_lookat([0, 0, 0])
            ctrl.set_up([0, 0, 1])
            ctrl.set_zoom(0.5)
            
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"\n❌ Erreur de visualisation: {e}")
            print("   Essayez d'installer Open3D avec: pip install --upgrade open3d")


def main():
    parser = argparse.ArgumentParser(description="Visualisation 3D des points LiDAR et détections")
    parser.add_argument('--h5', required=True, help='Fichier HDF5 d\'entrée')
    parser.add_argument('--csv', required=True, help='Fichier CSV de prédictions')
    parser.add_argument('--frame', type=int, default=0, help='Index de la frame à visualiser')
    parser.add_argument('--point-size', type=float, default=2.0, help='Taille des points')
    parser.add_argument('--width', type=int, default=1280, help='Largeur de la fenêtre')
    parser.add_argument('--height', type=int, default=720, help='Hauteur de la fenêtre')
    
    args = parser.parse_args()
    
    # Vérifier que les fichiers existent
    if not os.path.exists(args.h5):
        print(f"❌ Fichier HDF5 non trouvé: {args.h5}")
        return
    
    # Créer le visualiseur
    viz = LidarVisualizer()
    
    # Lancer la visualisation
    viz.visualize(
        h5_file=args.h5,
        csv_file=args.csv,
        frame_idx=args.frame,
        point_size=args.point_size,
        window_size=(args.width, args.height)
    )


if __name__ == "__main__":
    main()