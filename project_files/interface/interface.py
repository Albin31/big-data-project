from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5 import QtCore
from PyQt5.QtGui import *
import subprocess


class interface(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('projet.ui', self)
        self.Browse_bouton.clicked.connect(self.getFolder)
        self.launch_bouton.clicked.connect(self.execute_commande)
        self.connexion = "ssh hadoop@master "
        
        
    def getFolder(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open folder', 
        'c:\\')
        self.fichier_selec.setPlainText(fname)

    def execute_commande(self):
        print(self.choix_algo.currentText())
        chemin_dossier = self.fichier_selec.toPlainText()
        cmd_send1 = "scp "+chemin_dossier+"/train-labels.gz "+"hadoop@master:/data/"    
        cmd_send2 = "scp "+chemin_dossier+"/train-images.gz "+"hadoop@master:/data/"    
        
        cmd_comp = "./comp.sh"
        cmd_untar1 = "tar -xvzf data/train-labels.gz"
        cmd_untar2 = "tar -xvzf data/train-images.gz"
        cmd_generate = "java -cp ml.jar MnistGenerateFile data/train-labels data/train-images"        
        cmd_start = "./start.sh"
        cmd_copy = "./copy.sh"        
        cmd = "hdfs dfs -rm /output/*; hdfs dfs -rmdir /output"
        algo_choisi = self.choix_algo.currentText()
        if algo_choisi == "SVC":
            cmd_mod = "spark-submit --class SVC data/train-images data/train-labels --master spark://master:7077 ml.jar"
        elif algo_choisi == "Foret aléatoire":
            cmd_mod = "spark-submit --class RandomForest data/train-images data/train-labels --master spark://master:7077 ml.jar"
        elif algo_choisi == "Arbre de décision":
            cmd_mod = "spark-submit --class DecisionTree data/train-images data/train-labels --master spark://master:7077 ml.jar"
        else : 
            cmd_mod = "spark-submit --class Neural data/train-images data/train-labels --master spark://master:7077 ml.jar"
        cmd_stop = "./stop.sh"
        commande_generale = cmd_send1+";"+cmd_send2+";"+self.connexion+"\""+cmd_comp+";"+cmd_untar1+";"+cmd_untar2+";"+cmd_generate+";"+cmd_start+";"+cmd_copy+";"+cmd+";"+cmd_mod+";"+cmd_stop+"\""
        print(commande_generale)
        # process_general = subprocess.Popen(commande_generale.split(), stdout=subprocess.PIPE)
        # output, error = process_general.communicate()
        # print(output)


if __name__ == "__main__":
        app = QApplication([])
        ihm = interface() #on lance le programme principal
        ihm.show()
        app.exec_()