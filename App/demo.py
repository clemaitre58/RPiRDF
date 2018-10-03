import time
import picamera
import numpy as np
import RPi.GPIO as GPIO
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
        camera.resolution = (252, 252)
        camera.framerate = 24
        time.sleep(2)
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
    flag_start_descision = True


def process_start_learning(camera, l_individu):
    # TODO: Demander un texte si c'est nouvelle apprentissage ou si
    # TODO: ou si on a pas le label dans la basse

    # on récupère une image
    res = camera.resolution
    w = res[0]
    h = res[1]
    ind = np.empty((w, h, 3))
    camera.capture(ind, 'RGB')
    des_hu_col = hu_moment_color(ind)

    l_individu.append(des_hu_col)

    return l_individu


def process_stop_learning(l_individu, l_classe):
    # extract number of class
    Y = l_classe[:, 0]
    clf = SVC(gamma='auto', C=1000, random_state=42)
    clf.fit(l_individu, Y)
    # TODO: enregister 1- model 2- X 3- Y
    # Pour ne pas rentrer dans le process avant la prochaine interrupt sur
    # le bouton stop learning
    flag_stop_learning = False
    # repasse le flag de start à false pour ne pas continuer le calcul du Hu
    # dans la liste X
    flag_start_learning = False
    return clf


def process_start_decision(camera, clf):
    # TODO: faire une prédiction

    # on récupère une image
    res = camera.resolution
    w = res[0]
    h = res[1]
    ind = np.empty((w, h, 3))
    camera.capture(ind, 'RGB')
    des_hu_col = hu_moment_color(ind)
    classe_ind = clf.predict(des_hu_col)

    # TODO: faire l'annonce de la classe par la synthèse vocale


def process_stop_decision():
    # Pour arrêter la routine de décisison
    flag_start_descision = False
    # Pour rendre possible une future interruption "Start Desc"
    flag_stop_descision = False


if __name__ == '__main__':

    # initiation de la l'interruption
    camera = init()
    X = []
    # boucle infini = tache principale
    while True:
        # si une interruption c'est produite alors on lance le traitement c
        # adéquat
        if flag_start_learning is True:
            X, Y = process_start_learning(camera, X)
        if flag_stop_learning is True:
            model = process_stop_learning(X, Y)
        if flag_start_learning is True:
            process_start_decision(camera, model)
        if flag_stop_descision is True:
            process_stop_decision()

        pass
