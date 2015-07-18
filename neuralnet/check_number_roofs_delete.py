import experiment_settings as settings
from get_data import DataLoader
from os import listdir

def get_roofs(p):
    metal = 0
    thatch = 0
    for f in listdir(p):
        if f.startswith('000'):
            loader = DataLoader()
            xml_path = p+f[:-4]+'.xml'
            roofs = loader.get_roofs(xml_path, f)
            metalr = [1 for r in roofs if r.roof_type == 'metal']
            thatchr = [1 for r in roofs if r.roof_type == 'thatch']
            metal += sum(metalr)
            thatch += sum(thatchr)
    return metal, thatch



if __name__ == '__main__':
    train_p = settings.TRAINING_PATH
    validation_p = settings.VALIDATION_PATH
    testing_p = settings.TESTING_PATH
    for p in [train_p, validation_p, testing_p]:
        metal, thatch = get_roofs(p)
        print '{0}, metal {1}, thatch {2}'.format(p, metal, thatch)
