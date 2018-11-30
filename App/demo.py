import pygame
import os
import picamera
import joblib
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from DescGlob import hu_moment_color
from sklearn.svm import SVC
from skimage.io import imsave


# variable globale qui sera vue dans toutes les fonctions


class Manager:
    def __init__(self):
        self._isNew = bool()
        self._num_class = int()
        self._isSaveImage = bool()
        self._isSaveData = bool()
        self._isSaveModel = bool()
        self._isModelExist = bool()
        self._clf = None


class DataLearning():
    def __init__(self):
        self.l_individu = list()
        self.l_nom_classe = list()
        self._d_lut_nom = dict()


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


def process_start_learning(camera, data_learning, manager):
    # TODO: Demander un texte si c'est nouvelle apprentissage ou si
    # TODO: ou si on a pas le label dans la basse
    if data_learning._isNew:
        # On demande le nom de l'objet
        nom_obj = input("Quel est le nom de l'objet ?\n")
        manager._num_class += 1
        d_lut_nom[num_class] = nom_obj
        # Création d'un dossier
        if not os.path.exists(nom_obj):
            os.makedirs(nom_obj)
        isNew = False

    # on récupère une image
    res = camera.resolution
    w = res[0]
    h = res[1]
    ind = np.empty((w, h, 3), dtype=np.uint8)
    camera.capture(ind, 'rgb')

    des_hu_col = hu_moment_color(ind)

    data_learning._l_individu.append(des_hu_col)
    data_learning._l_nom_classe.append(manager._num_class)

    return data_learning, manger


def process_stop_learning(data_learning, manager):
    # extract number of class

    max_ech = len(l_classe)
    val_last_classe = data_learning._l_classe[max_ech-1]

    print('Last class :', val_last_classe)

    if val_last_classe == 1:
        print('Only one class. Impossible to learn something')
        global flag_stop_learning
        flag_stop_learning = False
        # TODO: change return with data and manger
        return 0, True, False

    Y = np.array(data._l_classe)
    X = np.array(data._l_individu)

    manager._clf = SVC(gamma='auto', C=1000, random_state=42)
    managet._clf.fit(X, Y)
    # Enristrement du modèle
    if manager._isSaveModel:
        joblib.dump(managet._clf, 'clf.joblib')

    if manager._isSaveData:
        # Enregistrement des données de chaque individu (descriptions)
        np.save('mat_desc.npy', data_learning._l_individu)
        # Enregistrement des données de description des classes
        np.save('mat_des_classe.npy', data_learning._l_classe)

    # Pour ne pas rentrer dans le process avant la prochaine interrupt sur
    # le bouton stop learning
    global flag_stop_learning
    flag_stop_learning = False
    # repasse le flag de start à false pour ne pas continuer le calcul du Hu
    # dans la liste X
    # on rend possible l'apprentissage d'une nouvelle classe la prochaine fois
    # qu'on va appuyer sur le bouton start learning
    manager._isNew = True
    manager._isModelExist = True
    return manager


def process_start_decision(camera, data_learning, manger):
    # on récupère une image
    if manager._isModelExist:
        res = camera.resolution
        w = res[0]
        h = res[1]
        ind = np.empty((w, h, 3), dtype=np.uint8)
        camera.capture(ind, 'rgb')
        des_hu_col = hu_moment_color(ind)
        classe_ind = clf.predict(des_hu_col)
        nom = data._d_lut_nom[classe_ind[0]]
        print('Object name : ', nom)
        # TODO: faire l'annonce de la classe par la synthèse vocale
        mes = 'flite -voice awb -t "' + nom + '"'
        os.system(mes)
        sleep(2)
    else:
        print("Aucun modèle existant, veuillez faire un apprentissage")


def process_stop_decision():
    # Pour arrêter la routine de décisison
    global flag_start_descision
    flag_start_descision = False
    # Pour rendre possible une future interruption "Start Desc"
    global flag_stop_descision
    flag_stop_descision = False


if __name__ == '__main__':

    # initialisation du manger et de la structure de donnée
    manager = Manager()
    data_learning = DataLearning()

    # initialisation du flag
    flag_stop_learning = False
    flag_start_learning = False
    flag_stop_descision = False
    flag_start_descision = False

    # initiation de la l'interruption

    camera = init()
    In=1
    pygame.init()
    w = 256
    h = 256
    size=(w,h)
    screen = pygame.display.set_mode(size)
    c = pygame.time.Clock() # create a clock object for timing

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
            data_learning, manager = process_start_learning(camera,
                                                            data_learning,
                                                            manager)
        if flag_stop_learning:
            manager = process_stop_learning(data_learning, manager)
            print('Learning done!')
        if flag_start_descision:
            process_start_decision(camera, data_learning, manager)
        if flag_stop_descision:
            process_stop_decision()
            print('stop decision')
        sleep(0.1)
        filename = str(In)+'.jpg' # ensure filename is correct
        camera.capture(filename)
        img=pygame.image.load(filename)
        screen.blit(img,(0,0))
        pygame.display.flip() # update the display
        c.tick(3) # only three images per second
        #In += 1



