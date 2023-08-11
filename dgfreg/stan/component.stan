data {
  int<lower=1> NY;
  int<lower=1> NC;
  int<lower=1> NG;
  matrix[NC, NY] S;
  matrix[NC, NG] G;
  vector[NY] y;
  int<lower=0,upper=1> likelihood;
  int<lower=1> N_train;
  int<lower=1> N_test;
  array[N_train] int<lower=1,upper=NY> ix_train;
  array[N_test] int<lower=1,upper=NY> ix_test;
}
transformed data {
  matrix[NY, NG] SGQ_ast = qr_thin_Q(S' * G) * sqrt(NY - 1);
  matrix[NG, NG] SGR_ast = qr_thin_R(S' * G) / sqrt(NY - 1);
  matrix[NG, NG] SGR_ast_inverse = generalized_inverse(SGR_ast);
  matrix[NY, NC] SQ_ast = qr_thin_Q(S') * sqrt(NY - 1);
  matrix[NC, NC] SR_ast = qr_thin_R(S') / sqrt(NY - 1);
  matrix[NC, NC] SR_ast_inverse = generalized_inverse(SR_ast);
}
parameters {
  vector[NC] dgfC_qr_param;
  vector[NG] dgfG_qr_param;
  real<lower=0> tauC;
  real muG;
  real<lower=0> tauG;
  real<lower=0> sigmaC;
}
transformed parameters {
  // use centred parameterisation in likelihood mode, non-centred in prior mode
  vector[NC] dgfC_qr = likelihood ? dgfC_qr_param : tauC * dgfC_qr_param;
  vector[NG] dgfG_qr = likelihood ? dgfG_qr_param : muG + tauG * dgfG_qr_param;
}
model {
  muG ~ normal(0, 2);
  tauG ~ normal(10, 2);
  tauC ~ normal(0, 2);
  sigmaC ~ normal(0, 2);
  if (likelihood){
    dgfC_qr_param ~ normal(0, tauC);
    dgfG_qr_param ~ normal(muG, tauG);
    y[ix_train] ~ normal((SGQ_ast * dgfG_qr + SQ_ast * dgfC_qr)[ix_train], sigmaC);
  }
  else {
    dgfC_qr_param ~ std_normal();
    dgfG_qr_param ~ std_normal();
  }
}
generated quantities {
  array[N_test] real llik;
  vector[NG] dgfG = SGR_ast_inverse * dgfG_qr;
  vector[NC] dgfC = G * dgfG + SR_ast_inverse * dgfC_qr;
  vector[N_test] yhat = (S' * dgfC)[ix_test];
  array[N_test] real yrep = normal_rng(yhat, sigmaC);
  real mae = mean(fabs(y[ix_test] - yhat));
  for (n in 1:N_test)
    llik[n] = normal_lpdf(y[ix_test[n]] | yhat[n], sigmaC);
}

