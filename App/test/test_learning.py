import os
import picamera
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from DescGlob import hu_moment_color
from sklearn.svm import SVC


# variable globale qui sera vue dans toutes les fonctions


def init():
    # configuration des broches en entree
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    # definition de l'interruption
    GPIO.add_event_detect(17, GPIO.FALLING, callback=press_btn,
                          bouncetime=300)
    GPIO.add_event_detect(27, GPIO.FALLING, callback=press_btn,
                          bouncetime=300)
    GPIO.add_event_detect(22, GPIO.FALLING, callback=press_btn,
                          bouncetime=300)
    GPIO.add_event_detect(23, GPIO.FALLING, callback=press_btn,
                          bouncetime=300)

    # démarrage de picaméra
    # paramétrage de la cam
    camera = picamera.PiCamera()
    camera.resolution = (256, 256)
    camera.framerate = 24
    sleep(2)
    print(camera.resolution)

    return camera


def press_btn(channel):
    # function qui sera appelé lorsque le programme sur interrompu
    print(channel)
    if channel == 17:
        global flag_start_learning
        flag_start_learning = True
    elif channel == 27:
        global flag_stop_learning
        flag_stop_learning = True
        global flag_start_learning
        flag_start_learning = False
    elif channel == 22:
        global flag_start_descision
        flag_start_descision = True
    elif channel == 23:
        # if flag_start_learning is not True:
        global flag_stop_descision
        flag_stop_descision = True


def process_start_learning(camera, l_individu, l_nom_classe, d_lut_nom,
                           isNew, num_class):
    # TODO: Demander un texte si c'est nouvelle apprentissage ou si
    # TODO: ou si on a pas le label dans la basse
    if isNew is True:
        # On demande le nom de l'objet
        nom_obj = input("Quel est le nom de l'objet")
        num_class += 1
        d_lut_nom[num_class] = nom_obj
        isNew = False

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

    # initialisation du flag
    flag_stop_learning = False
    flag_start_learning = False
    flag_stop_descision = False
    flag_start_descision = False

    # initiation de la l'interruption

    camera = init()
    print(camera.resolution)
    print('PiRDF start')
    X = []
    Y = []
    d_lut_nom = {}
    isNew = True
    isModelExist = False
    num_class = 0

    # boucle infini = tache principale
    while True:
        # si une interruption c'est produite alors on lance le traitement c
        # adéquat
        if flag_start_learning:
            X, Y, d_lut_nom,
            isNew, num_class = process_start_learning(camera,
                                                      X,
                                                      Y,
                                                      d_lut_nom,
                                                      isNew,
                                                      num_class)
        if flag_stop_learning:
            print('stop learning')
        if flag_start_descision:
            print('start decision')
        if flag_stop_descision:
            print('stop decision')
        sleep(0.1)
