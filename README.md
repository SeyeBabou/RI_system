# Binôme 
 
**El hadji babou SEYE**


# Processus à suivre


## 1. Faire un clone du projet
```bash
git clone https://github.com/Babou001/RI_system.git
```
## 2. Installer un environnement
```bash
python3 -m venv myenv
```
## 3. Activer l'environment
### Sous linux/ios
```bash
source myenv/bin/activate
```
### Sous windows
```bash
myenv\Scripts\activate
```
## 4. Installer les dépendances
```bash
pip install -r requirements.txt
```
## 5. Installer cette version de numpy
On a décidé de l'installer après vu qu'il pose des problème de conflit. Vous pouvez voir une erreur ou un avertissement mais cela n'empêche pas à l'application de fonctionner normalement
```bash
 pip install numpy==1.23.5
```

## 6. Lancer la classe interface
```bash
python3 interface.py
```
Le lancement de l'application peut prendre une à deux minutes. Cela dépend des ordinateurs.

# Etat du projet
Pour ce qui est de ce projet , il nous reste l'auto complétion à faire et le système d'évaluation mais nous avons pu faire tout le reste