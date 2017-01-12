
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

CYCLE_IDX                 (idx, 1)        = 30 ;

TOT_CPU_TIME              (idx, 1)        =  30.5 ;

TEST_VAL                  (idx, [1:   4]) = [  1.23456E+02 0.00010  2.23456E+02 0.00012 ];
TEST_MAT                  (idx, [1:   8]) = [  1.23456E+02 0.00010  2.23456E+02 0.00012  3.23456E+02 0.00014  4.23456E+02 0.00016  ];
