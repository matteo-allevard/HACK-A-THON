# explore_h5.py
import h5py
import numpy as np

def explore_h5_file(file_path):
    """Explorer la structure d'un fichier HDF5"""
    print(f"\nExploration du fichier: {file_path}")
    print("=" * 50)
    
    with h5py.File(file_path, 'r') as f:
        # Afficher toutes les clés (datasets) disponibles
        print("Clés disponibles dans le fichier:")
        for key in f.keys():
            print(f"  - {key}")
            
            # Afficher des infos sur chaque dataset
            dataset = f[key]
            print(f"    Shape: {dataset.shape}")
            print(f"    Type: {dataset.dtype}")
            print(f"    Exemple (premiers éléments): {dataset[:5]}")
            print()
        
        # Vérifier la taille totale
        if len(f.keys()) > 0:
            first_key = list(f.keys())[0]
            print(f"\nTaille totale du dataset principal: {len(f[first_key])} éléments")

if __name__ == "__main__":
    # Mettez le chemin vers un de vos fichiers .h5
    h5_file = "./airbus_hackathon_trainingdata/scene_1.h5"  # À modifier avec le bon chemin
    explore_h5_file(h5_file)