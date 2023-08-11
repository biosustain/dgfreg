data {
  int<lower=1> NY;
  int<lower=1> NC;
  matrix[NC, NY] S;
  vector[NY] y;
  int<lower=0,upper=1> likelihood;
  int<lower=1> Ntrain;
  int<lower=1> Ntest;
  array[Ntrain] int<lower=1,upper=NY> ix_train;
  array[Ntest] int<lower=1,upper=NY> ix_test;
}
transformed data {
  matrix[NY, NC] SQ_ast = qr_thin_Q(S') * sqrt(NY - 1);
  matrix[NC, NC] SR_ast = qr_thin_R(S') / sqrt(NY - 1);
  matrix[NC, NC] SR_ast_inverse = generalized_inverse(SR_ast);
}
parameters {
  real muC;
  real<lower=0> tauC;
  real<lower=0> sigmaC;
  vector[NC] dgfC_qr_param;
}
transformed parameters {
  // use centred parameterisation in likelihood mode, non-centred in prior mode
  vector[NC] dgfC_qr = likelihood ? dgfC_qr_param : muC + tauC * dgfC_qr_param;
}
model {
  muC ~ normal(0, 2);
  tauC ~ normal(1, 3);
  sigmaC ~ normal(0, 2);
  if (likelihood){
    dgfC_qr_param ~ normal(muC, tauC);
    y[ix_train] ~ normal((SQ_ast * dgfC_qr)[ix_train], sigmaC);
  }
  else {
    dgfC_qr_param ~ std_normal();
  }
}
generated quantities {
  array[Ntest] real yrep;
  array[Ntest] real llik;
  vector[NC] dgfC = SR_ast_inverse * dgfC_qr;
  vector[Ntest] yhat = (S' * dgfC)[ix_test];
  real mae = mean(fabs(y[ix_test] - yhat));
  {
    yrep = normal_rng(yhat, sigmaC);
    for (n in 1:Ntest) llik[n] = normal_lpdf(y[ix_test[n]] | yhat[n], sigmaC);
  }
}
