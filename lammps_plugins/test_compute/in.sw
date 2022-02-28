units		metal
atom_style	atomic

lattice		diamond 5.431
region		box block 0 $L 0 $L 0 $L
create_box	1 box
create_atoms	1 box

newton on
pair_style	sw
pair_coeff	* * Si.sw Si
mass            1 28.06

velocity	all create 1000.0 376847 loop geom

neighbor	1.0 bin
neigh_modify    delay 5 every 1

fix		1 all nve

timestep	0.001

#displace_atoms all random 0.1 0.1 0.1 654321

compute peatom all pe/atom
compute stressatom all stress/atom NULL virial

#dump 1 all custom 10 si.dump id type x y z fx fy fz c_peatom c_stressatom[*]
#dump_modify 1 sort id

thermo 1000
thermo_style custom step temp pe etotal press pxx pyy pzz pxy pxz pyz spcpu

if "$L > 5" then "variable nsteps equal 100" else "variable nsteps equal 1000"
#variable nsteps equal 2000
run		${nsteps}
