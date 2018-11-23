import os
import picamera
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from DescGlob import hu_moment_color
from sklearn.svm import SVC


# variable globale qui sera vue dans toutes les fonctions


def init():
    global flag_stop_learning
    global flag_start_learning
    global flag_stop_descision
    global flag_start_descision

    # configuration de la broche 7 en entree
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(11, GPIO.IN)
    GPIO.setup(13, GPIO.IN)
    GPIO.setup(15, GPIO.IN)
    GPIO.setup(16, GPIO.IN)
    # definition de l'interruption
    GPIO.add_event_detect(11, GPIO.RISING, callback=stop_learning,
                          bouncetime=300)
    GPIO.add_event_detect(13, GPIO.RISING, callback=start_learning,
                          bouncetime=300)
    GPIO.add_event_detect(15, GPIO.RISING, callback=stop_descision,
                          bouncetime=300)
    GPIO.add_event_detect(16, GPIO.RISING, callback=start_descision,
                          bouncetime=300)
    # initialisation du flag
    flag_stop_learning = False
    flag_start_learning = False
    flag_stop_descision = False
    flag_start_descision = False

    # démarrage de picaméra
    # paramétrage de la cam
    with picamera.Picamera() as camera:
        camera.resolution = (256, 256)
        camera.framerate = 24
        sleep(2)
        return camera


def stop_learning():
    # function qui sera appelé lorsque le programme sur interrompu
    flag_stop_learning = True


def start_learning():
    # function qui sera appelé lorsque le programme sur interrompu
    flag_start_learning = True


def stop_descision():
    # function qui sera appelé lorsque le programme sur interrompu
    flag_stop_descision = True


def start_descision():
    # function qui sera appelé lorsque le programme sur interrompu
    # on ne considère l'enterruption uniquement si on n'est pas phase
    # d'apprentissage
    if flag_start_learning is not True:
        flag_start_descision = True


def process_start_learning(camera, l_individu, l_nom_classe, d_lut_nom,
                           isNew, num_class):
    # TODO: Demander un texte si c'est nouvelle apprentissage ou si
    # TODO: ou si on a pas le label dans la basse


    # on récupère une image
    res = camera.resolution
    w = res[0]
    h = res[1]
    ind = np.empty((w, h, 3))
    camera.capture(ind, 'rgb')
    des_hu_col = hu_moment_color(ind)

    l_individu.append(des_hu_col)
    l_nom_classe.append(num_class)

    return l_individu, l_nom_classe, d_lut_nom, isNew, num_class


def process_stop_learning(l_individu, l_classe, isNew, isModelExist):
    # extract number of class

    Y = np.array(l_classe)
    X = np.array(l_individu)
    clf = SVC(gamma='auto', C=1000, random_state=42)
    clf.fit(X, Y)
    # Enregistrement des données de chaque individu (descriptions)
    np.save('mat_desc.npy', l_individu)
    # Enregistrement des données de description des classes
    np.save('mat_des_classe.npy', l_classe)
    # Pour ne pas rentrer dans le process avant la prochaine interrupt sur
    # le bouton stop learning
    flag_stop_learning = False
    # repasse le flag de start à false pour ne pas continuer le calcul du Hu
    # dans la liste X
    flag_start_learning = False
    # on rend possible l'apprentissage d'une nouvelle classe la prochaine fois
    # qu'on va appuyer sur le bouton start learning
    isNew = True
    isModelExist = True
    return clf, isNew, isModelExist


def process_start_decision(camera, clf, d_lut_nom, isModelExist):
    # on récupère une image
    if isModelExist is True:
        res = camera.resolution
        w = res[0]
        h = res[1]
        ind = np.empty((w, h, 3))
        camera.capture(ind, 'RGB')
        des_hu_col = hu_moment_color(ind)
        classe_ind = clf.predict(des_hu_col)
        nom = d_lut_nom[classe_ind]
        # TODO: faire l'annonce de la classe par la synthèse vocale
        mes = 'flite -voice awb -t "' + nom + '"'
        os.system(mes)
        sleep(0.1)
    else:
        print("Aucun modèle existant, veuillez faire ")


def process_stop_decision():
    # Pour arrêter la routine de décisison
    flag_start_descision = False
    # Pour rendre possible une future interruption "Start Desc"
    flag_stop_descision = False


if __name__ == '__main__':

    # initiation de la l'interruption

    cam = init()

    # boucle infini = tache principale
    while True:
        # si une interruption c'est produite alors on lance le traitement c
        # adéquat
        if flag_start_learning is True:
            print('start learning')
        if flag_stop_learning is True:
            print('stop learning')
        if flag_start_descision is True:
            print('start decision')
        if flag_stop_descision is True:
            print('stop decision')

        pass
