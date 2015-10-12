package com.simiacryptus.mindseye.training;

public interface RateTrainingComponent extends TrainingComponent {

  double getRate();

  RateTrainingComponent setRate(double rate);

}
