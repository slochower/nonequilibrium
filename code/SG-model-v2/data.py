import glob as glob

dir = './pka-md-data/'
unbound_files = sorted(glob.glob(dir + 'apo/' + '*'))
bound_files = sorted(glob.glob(dir + 'atpmg/' + '*'))
