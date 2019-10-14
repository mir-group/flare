import os, shutil, sys

if "test_fail" in sys.argv:
	exit(0)

curr_path = os.path.dirname(os.path.abspath(__file__))
vasprun_name = os.path.join(curr_path, 'test_vasprun.xml')
poscar_name = os.path.join(curr_path, 'test_POSCAR')
calc_dir = os.path.join(curr_path, '..')

shutil.copyfile(poscar_name, os.path.join(calc_dir, "POSCAR"))
shutil.copyfile(vasprun_name, os.path.join(calc_dir, "vasprun.xml"))
