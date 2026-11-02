producer: L2Norm -> cumsum -> KK.T -> solve_tri -> recompute_wu -> W,U, g_sum
consumer: W, U -> fwdh -> v_new, h -> fwdo
while B * H <= total_core_num / 2;

for each stage, refer to relevant skills, must using cv-fusion and regbase speed up.
for crosscore sync: using crosscoresetflag/crosscorewaitflag<mode 0x0>.
for each minimum exec unit, using a module or single function to realization for debug and combination;
re-built ub&workspace&l1 memory allocation for this scenary.
