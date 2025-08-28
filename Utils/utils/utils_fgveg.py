import os
import glob

def get_matching_files(folder1, folder2, pattern="*/*"):
    """
    Vérifie la correspondance des fichiers entre deux dossiers et retourne les fichiers correspondants.

    Args:
        folder1 (str): Chemin du premier dossier.
        folder2 (str): Chemin du deuxième dossier.
        pattern (str, optional): Motif pour `glob.glob` (par défaut "**/*" pour inclure tous les sous-dossiers).

    Returns:
        list, list: Deux listes contenant les chemins des fichiers correspondants.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}  # Extensions valides

    def normalize_filename(filepath, base_folder):
        """Renvoie le nom du fichier sans extension en conservant le chemin relatif"""
        rel_path = os.path.relpath(filepath, base_folder)  # Ex: "Bois1/image1.jpg" devient "Bois1/image1"
        return os.path.splitext(rel_path)[0]

    # Récupérer tous les fichiers d'images
    files1 = sorted(glob.glob(os.path.join(folder1, pattern), recursive=True))
    files2 = sorted(glob.glob(os.path.join(folder2, pattern), recursive=True))

    # Filtrer les fichiers d'images valides
    files1 = [p for p in files1 if os.path.splitext(p)[1].lower() in valid_extensions]
    files2 = [p for p in files2 if os.path.splitext(p)[1].lower() in valid_extensions]

    # Stocker les fichiers du dossier 2 dans un dictionnaire {nom_sans_extension: chemin_complet}
    file_dict2 = {normalize_filename(p, folder2): p for p in files2}

    matched_files1 = []
    matched_files2 = []

    missing_in_folder2 = []  # Fichiers présents dans folder1 mais pas dans folder2

    # Vérifier que chaque fichier du dossier 1 a un match dans le dossier 2
    for file1 in files1:
        norm_name1 = normalize_filename(file1, folder1)  # Nom relatif sans extension

        if norm_name1 in file_dict2:  # Vérifier si un fichier correspondant existe
            matched_files1.append(file1)
            matched_files2.append(file_dict2[norm_name1])
        else:
            missing_in_folder2.append(norm_name1)  # Ajouter aux fichiers manquants

    # Identifier les fichiers présents dans folder2 mais pas dans folder1
    missing_in_folder1 = set(file_dict2.keys()) - {normalize_filename(p, folder1) for p in files1}

    #Debug
    # common_files = matched_files1 & matched_files2  # Intersection des fichiers communs
    # print(f"COMMUNS::: {len(common_files)}")
    if missing_in_folder2:
        print(f"{len(missing_in_folder2)} fichiers manquants dans '{folder2}': {missing_in_folder2}")

    if missing_in_folder1:
        print(f"{len(missing_in_folder1)} fichiers manquants dans '{folder1}': {missing_in_folder1}")

    return matched_files1, matched_files2