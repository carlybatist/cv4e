print("hello")
import numpy as np
import librosa
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.annotations import categorical_to_one_hot

#making alias for connecting to remote server
    #need to make sure you're in home folder (code= cd ~ )
#make hidden config file
    #mkdir .ssh
#open folder
    #nano .ssh/config
    #Host [name of server you want to access]
    #HostName [host name/IP address]
    #User (user you use to access server)
#press Cntrl O, enter, Cntrl X

#View --> Command Palette --> Remote-SSH: Connect to Host
