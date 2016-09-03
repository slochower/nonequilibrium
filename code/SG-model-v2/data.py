import glob as glob

dir = '../../md-data/pka-md-data/'
unbound_files = sorted(glob.glob(dir + 'apo/' + '*'))
bound_files = sorted(glob.glob(dir + 'atpmg/' + '*'))
