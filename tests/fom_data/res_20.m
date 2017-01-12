
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

CYCLE_IDX                 (idx, 1)        = 20 ;

TOT_CPU_TIME              (idx, 1)        =  20.5 ;

TEST_VAL                  (idx, [1:   4]) = [  1.23456E+02 0.00020  2.23456E+02 0.00022 ];
TEST_MAT                  (idx, [1:   8]) = [  1.23456E+02 0.00020  2.23456E+02 0.00022  3.23456E+02 0.00024  4.23456E+02 0.00026  ];
