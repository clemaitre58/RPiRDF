import RPi.GPIO as GPIO


# variable globale qui sera vue dans toutes les fonctions
flag_stop_learning = False
flag_start_learning = False
flag_stop_descision = False
flag_start_descision = False


def init():
    # configuration de la broche 7 en entree
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(7, GPIO.IN)
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


def process_start_learning():
    flag_start_learning = False


def process_stop_learning():
    flag_stop_learning = False


def process_start_decision():
    flag_start_descision = False


def process_stop_decision():
    flag_stop_descision = False


if __name__ == '__main__':

    # 1- initiation de la l'interruption
    init()
    # 2- boucle infini = tache principale
    while True:
        # 3- si une interruption c'est produite alors on lance le traitement c
        # adéquat
        if flag_start_learning is True:
            process_start_learning()
        if flag_stop_learning is True:
            process_stop_learning()
        if flag_start_learning is True:
            process_start_decision()
        if flag_stop_descision is True:
            process_stop_decision()

        pass
