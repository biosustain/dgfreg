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
  matrix[NR, NG] Qstar = qr_thin_Q(S' * G) * sqrt(NR - 1);
  matrix[NG, NG] Rstar = qr_thin_R(S' * G) / sqrt(NR - 1);
  matrix[NG, NG] Rstar_inverse = generalized_inverse(Rstar);
  vector[NR] sqrt_nobs = sqrt(nobs);
}
parameters {
  vector[NC] qC;
  vector[NG] qG;
  real<lower=0> tauC;
  real<lower=0> sigma;
}
transformed parameters {
  vector[NG] dgfG = Rstar_inverse * qG;
}
model {
  dgfG ~ normal(-500, 1000);
  tauC ~ normal(0, 4);
  sigma ~ normal(0, 4);
  qC ~ normal(0, tauC);
  if (likelihood){
    y[ix_train] ~ normal((Qstar * qG + S' * qC)[ix_train], sigma / sqrt_nobs[ix_train]);
  }
}
generated quantities {
  array[N_test] real llik;
  vector[NC] dgfC = G * dgfG + qC;
  vector[NR] dgr = (S' * dgfC);
  array[N_test] real yrep = normal_rng(dgr[ix_test], sigma / sqrt_nobs[ix_test]);
  real mae = mean(abs(y[ix_test] - dgr[ix_test]));
  for (n in 1:N_test)
    llik[n] = normal_lpdf(y[ix_test[n]] | dgr[n], sigma / sqrt_nobs[ix_test[n]]);
}


