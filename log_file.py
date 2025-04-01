import logging



def phase_1(acc):
    log = logging.getLogger(acc)
    log.setLevel("DEBUG") # DEBUG | Error | Critical | info | warning

    # Create a file handler for the script
    handler = logging.FileHandler(f"C:\\Users\\DELL\\Downloads\\Credit_Card_Project\\file_loggings\\{acc}.log", mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log

