import os, sys


def train_model(txt_rddf, image_folder):
    dados_rddf = []
    if(image_folder[-1] == '/'):
        image_folder = image_folder[:-1]
    with open(txt_rddf, 'r') as f:
        dados_rddf = [line.strip().split(" ") for line in f]

    for i in range(len(dados_rddf)):
        if(os.path.isfile(image_folder + '/' + dados_rddf[4] + '-r.png')):
            print('existe')
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')

#train_model("/dados/rddf_predict/te.txt", "/dados/log_png_1003")
train_model(sys.argv[1], sys.argv[2])
