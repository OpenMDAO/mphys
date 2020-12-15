import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

f = open('mesh_flatplate_65x65.su2', 'r')

line = f.readline()
xpts = []
ct = 0
while line:

    l = line.split()
    if is_number(l[0]) and is_number(l[1]):
        if float(l[1]) == 0.0 and ct >= 50 and ct<8328:
            xpts.append(float(l[0]))

    line = f.readline()
    ct += 1

f.close()

yvals = -1*np.linspace(0.0, 1.0, 11)
pts = []
xvals = []
xpts = np.linspace(0.0, 0.3048, 30)
for x in xpts:

    if abs(x) < 1e-10:
        x = 0.0

    if x >= 0.0:
        xvals.append(x)
        for y in yvals:
            pts.append([x, y])

f = open('plate_solid.su2', 'w')

plate_inner = []
plate_core = []
plate_edge = []
all_elems = []

nodes = np.arange(0, len(xvals)*len(yvals), dtype=np.int).reshape(len(xvals), len(yvals))

f.write('NPOIN= ' + str(len(nodes.flatten())) + '\n')
# write all nodes to file in su2 format
for i in range(len(xvals)):
    for j in range(len(yvals)):
        f.write(str(xvals[i]) + ' ' + str(yvals[j]) + ' ' + str(nodes[i, j]) + '\n')

# write all quads to file in su2 format
el_ct = (nodes.shape[1]-1)*(nodes.shape[0]-1)
f.write('NELEM= ' + str(el_ct) + '\n')
for j in range(0, nodes.shape[1]-1, 1):
    for i in range(0, nodes.shape[0]-1, 1):

        f.write('9 ' + str(nodes[i,j]) + ' ' + str(nodes[i,j+1]) + ' ' + str(nodes[i+1,j+1]) + ' ' + str(nodes[i+1,j]) + '\n')

# now do all the markers
f.write('NMARK= 4\n')
f.write('MARKER_TAG= plate_inner\n')
inner_ct = nodes.shape[0] - 1
f.write('MARKER_ELEMS= ' + str(inner_ct) + '\n')
for i in range(nodes.shape[0] - 1):
    f.write('3 ' + str(nodes[i,0]) + ' ' + str(nodes[i+1,0]) + '\n')

f.write('MARKER_TAG= plate_edge\n')
edge_ct = (nodes.shape[1] - 1) * 2
f.write('MARKER_ELEMS= ' + str(edge_ct) + '\n')
for i in range(nodes.shape[1] - 1):
    f.write('3 ' + str(nodes[0, i]) + ' ' + str(nodes[0, i+1]) + '\n')
for i in range(nodes.shape[1] - 1):
    f.write('3 ' + str(nodes[nodes.shape[0]-1, i]) + ' ' + str(nodes[nodes.shape[0]-1, i+1]) + '\n')

f.write('MARKER_TAG= plate_core\n')
f.write('MARKER_ELEMS= ' + str(inner_ct) + '\n')
for i in range(nodes.shape[0] - 1):
    f.write('3 ' + str(nodes[i,nodes.shape[1]-1]) + ' ' + str(nodes[i+1,nodes.shape[1]-1]) + '\n')
