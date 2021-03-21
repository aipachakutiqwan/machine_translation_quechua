
from utils import read_txt_file, write_excel
import pandas as pd

class PreprocessData:

    def __init__(self):
        return None

    def preprocess(self, path_esp, path_que, path_out):
        txt_esp = read_txt_file(path_esp)
        txt_que = read_txt_file(path_que)
        splitted_txt_esp = txt_esp.split('\n')
        splitted_txt_que = txt_que.split('\n')
        print(len(splitted_txt_esp), len(splitted_txt_que))
        major_length = len(splitted_txt_esp) if len(splitted_txt_esp)>len(splitted_txt_que) else len(splitted_txt_que)
        dic_merged = {"id": list(), "quechua": list(), "spanish": list()}

        for i in range(0, major_length):
            print("-----------------------------------------------------------------------------------")
            if i < len(splitted_txt_que):
                que = splitted_txt_que[i]+'\n'
            else:
                que = ""
            
            if i < len(splitted_txt_esp):
                esp = splitted_txt_esp[i]+'\n'
            else:
                esp = ""

            dic_merged['id'].append(i)
            dic_merged['quechua'].append(que)
            dic_merged['spanish'].append(esp)
            print(que,"  >>>>>>>>>>>" ,esp)
        
        dataframe = pd.DataFrame(dic_merged)
        write_excel(dataframe, path_out)
        


if __name__ == "__main__":

    path_esp = "/Users/c325018/Documents/TraductorQuechua/data/principito/ESPrincipito.txt"
    path_que = "/Users/c325018/Documents/TraductorQuechua/data/principito/QUPrincipito.txt"
    path_out = "/Users/c325018/Documents/TraductorQuechua/data/principito/little_prince_merged_QUE_ESP.csv"

    PD = PreprocessData()
    PD.preprocess(path_esp, path_que, path_out)
    