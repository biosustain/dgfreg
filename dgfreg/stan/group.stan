data {
  int<lower=1> NR;
  int<lower=1> NC;
  int<lower=1> NG;
  matrix[NC, NR] S;
  matrix[NC, NG] G;
  vector[NR] y;
  vector[NR] nobs;
  int<lower=0,upper=1> likelihood;
  int<lower=1> N_train;
  int<lower=1> N_test;
  array[N_train] int<lower=1,upper=NR> ix_train;
  array[N_test] int<lower=1,upper=NR> ix_test;
}
transformed data {
  matrix[NY, NG] SGQ_ast = qr_thin_Q(S' * G) * sqrt(NY - 1);
  matrix[NG, NG] SGR_ast = qr_thin_R(S' * G) / sqrt(NY - 1);
  matrix[NG, NG] SGR_ast_inverse = generalized_inverse(SGR_ast);
  vector[NR] sqrt_nobs = sqrt(nobs);
}
parameters {
  vector[NG] qG;
  real<lower=0> sigmaC;
}
model {
  sigmaC ~ normal(0, 2);
  if (likelihood){
    y[ix_train] ~ normal((SGQ_ast * dgfG_qr)[ix_train], sigmaC / sqrt_nobs[ix_train]);
  }
}
generated quantities {
  array[N_test] real llik;
  vector[NG] dgfG = SGR_ast_inverse * qG;
  vector[NC] dgfC = G * dgfG + qC;
  vector[NR] dgr = (S' * dgfC);
  array[N_test] real yrep = normal_rng(dgr[ix_test], sigma / sqrt_nobs[ix_train]);
  real mae = mean(abs(y[ix_test] - dgr[ix_test]));
  for (n in 1:N_test)
    llik[n] = normal_lpdf(y[ix_test[n]] | dgr[n], sigma / sqrt_nobs[ix_test[n]]);
}
