from pipython.pidevice.interfaces.pisocket import PISocket
from pipython import GCSDevice

gateway = PISocket(host=ip)
device = GCSDevice(gateway=gateway)
print(device.pidevice.qIDN())

device.gcsdevice.qPOS() # read position
## output: ordered dict:
#  OrderedDict([('1', 6.929430485), ('2', 100.1199341), ('3', -26.24203491)])
#               Pist [um]           rotX [urad]         rotY [urad]

device.gcsdevice.MOV(ordered_dict) # set shape

cmask = 'E-727.3SDA'

import numpy as np
from pipython.pidevice.interfaces.pisocket import PISocket
from pipython import GCSDevice

ips = [f'192.168.29.15{x}' for x in range(1,7)]
 
class petalDM():

    def __init__(self):
        self._gateways = [PISocket(host=ip) for ip in ips]
        self._devices = [GCSDevice(gateway=gateway).gcsdevice for gateway in self._gateways]

        if not all(
            [dev.connected for dev in self._devices]
        ):
            raise RuntimeError('Some connaction did not get estrablished')

    def get_shape(self):
        pos = []
        for i,dev in enumerate(self._devices):
            posx = dev.qPOS()
            posx = [posx['1'], posx['2'], posx['3']]
            pos.extend(posx)
        return np.asarray(pos)

    def set_shape(self, cmd, differential=False):
        if not len(cmd) == 18:
            raise ValueError('command lenght must be 18')
        if differential:
            cmd += self.get_shape()
        seg_cmds = []
        for k in range(6):
            seg_cmds.append(cmd[k*3:k*3+3])
            print(k, cmd[k*3:k*3+3])
        for scmd,dev in zip(seg_cmds,self._devices):
            odict = {'1': scmd[0],'2':scmd[1], '3':scmd[2]}
            dev.MOV(odict)
            dev.checkerror()
    
    def close_connection(self):
        for dev in self._devices:
            dev.close()

    def connect(self):
        self._devices = [GCSDevice(gateway=gateway).gcsdevice for gateway in self._gateways]




