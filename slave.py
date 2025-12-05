## Create the S2M matrix for slaving actuators on-spot
ffwd = dm.ff

sid = np.array([41, 22, 21, 2, 1, 0, 111, 112, 113, 132, 133, 152]) # slave ids
mid = np.arange(222) # master ids
mid[sid] = 1e6
mid = np.array([x for x in mid if x != 1e6])

# slave ffwd
temp = dm.ffwd[:,sid]
k11 = temp[sid,:]

# m-s ffwd
temp = dm.ff[:,mid]
k01 = temp[sid,:]

# master 2 slave matrix
s2m = -np.linalg.pinv(k11) * k10

cmd # a command 
cmd[sid] = s2m @ cmd[mid]

## compute new IM with slaved actuators

# for one shell only, or attach the ff in blocks and do it for all

ff = dm.ff
u, w, vt = np.linalg.svd(ff)

IM = flattening.IM # get the interaction matrix from the flattening class

ZIM = vt.T @ IM # [:111] per una shell

# compute new Vt matrix slaved

temp = vt.T[mid,:] # mid [105] per singola shell
nv = temp[:,mid]

# new interaction matrix
nIM = nv.T @ ZIM[mid,:]

# image to check
img = np.ma.masked_array(int_cube_img, mask=analysisMask)
img[img.mask==0] = nIM[modo,:] # modo è dei master, quindi nActs - nSlaved
