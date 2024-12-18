# build.py
import PyInstaller.__main__
import os
import sys

def create_optimized_exe():
    # Définir le nom du script principal
    script_name = "Poker-Range-Extraction.py"
    
    # Définir les options d'optimisation pour PyInstaller
    options = [
        script_name,
        '--noconfirm',  # Remplace les fichiers existants
        '--onefile',    # Crée un seul fichier exe
        '--noconsole',  # Pas de console
        '--clean',      # Nettoie avant la compilation
        
        # Dépendances obligatoires
        '--hidden-import', 'sklearn',
        '--hidden-import', 'sklearn.cluster',
        '--hidden-import', 'sklearn.cluster._kmeans',
        '--hidden-import', 'sklearn.utils',
        '--hidden-import', 'sklearn.utils.murmurhash',
        '--hidden-import', 'sklearn.utils._cython_blas',
        '--hidden-import', 'sklearn.utils._typedefs',
        '--hidden-import', 'sklearn.utils._heap',
        '--hidden-import', 'sklearn.utils._sorting',
        '--hidden-import', 'sklearn.utils._vector_sentinel',
        '--hidden-import', 'sklearn.utils.sparsefuncs_fast',
        '--hidden-import', 'sklearn.utils._random',
        '--hidden-import', 'sklearn.utils.validation',
        '--hidden-import', 'sklearn.utils.fixes',
        '--hidden-import', 'scipy',
        '--hidden-import', 'scipy.sparse',
        '--hidden-import', 'scipy.sparse._csr',
        '--hidden-import', 'scipy.sparse.csr',
        '--hidden-import', 'scipy.sparse.data',
        '--hidden-import', 'numpy',
        
        # Collect all submodules
        '--collect-submodules', 'sklearn',
        '--collect-submodules', 'scipy',
        '--collect-submodules', 'numpy',
        
        # Ajouter des data nécessaires
        '--collect-data', 'sklearn',
        '--collect-data', 'scipy',
        '--collect-data', 'numpy',
        
        # Spécifier le dossier de travail
        '--workpath', './build',
        '--distpath', './dist',
        
        # Mode fenêtré
        '-w'
    ]
    
    print("Début de la compilation...")
    PyInstaller.__main__.run(options)
    print("Compilation terminée !")

if __name__ == "__main__":
    # Créer les dossiers si nécessaire
    os.makedirs('./build', exist_ok=True)
    os.makedirs('./dist', exist_ok=True)
    create_optimized_exe()
